import numpy as np
import tqdm
import os
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.io import update_config
from keypoint_moseq.util import find_matching_videos, get_edges

def sample_error_frames(confidences, bodyparts, use_bodyparts,
                        num_bins=10, num_samples=100, num_videos=10,
                        conf_pseudocount=1e-3):
    """
    Randomly sample frames, enriching for those with low confidence 
    keypoint detections.

    Parameters
    ----------
    confidences: dict
        Keypoint detection confidences for a collection of sessions 

    bodyparts: list
        Label for each keypoint represented in ``confidences``

    use_bodyparts: list
        Ordered subset of keypoint labels to be used for modeling

    num_bins: int, default=10
        Number of bins to use for enriching low-confidence keypoint
        detections. Confidence values for all used keypoints are 
        divided into log-spaced bins and an equal number of instances
        are sampled from each bin.

    num_samples: int, default=100
        Total number of frames to sample

    num_videos: int, default=10
        Maximum number of videos to use. Fewer videos helps with
        faster frame loading.

    conf_pseudocount: float, default=1e-3
        Pseudocount used to augment keypoint confidences.
        
    Returns
    -------
    sample_keys: list of tuples
        List of sampled frames as tuples with format 
        (key, frame_number, bodypart)

    """
    confidences = {k:v+conf_pseudocount for k,v in confidences.items()}
    all_videos = sorted(confidences.keys())
    num_videos = min(num_videos, len(all_videos))
    use_videos = np.random.choice(all_videos, num_videos, replace=False)
    
    all_confs = np.concatenate([v.flatten() for v in confidences.values()])
    min_conf, max_conf = np.nanmin(all_confs), np.nanmax(all_confs)
    thresholds = np.logspace(np.log10(min_conf), np.log10(max_conf), num_bins)
    mask = np.array([bp in use_bodyparts for bp in bodyparts])[None,:]
        
    sample_keys = []
    for low,high in zip(thresholds[:-1],thresholds[1:]):
        
        samples_in_bin = []
        for key,confs in confidences.items():
            for t,k in zip(*np.nonzero((confs>=low)*(confs<high)*mask)): 
                samples_in_bin.append((key,t,bodyparts[k]))
        
        if len(samples_in_bin)>0:
            n = min(num_samples//num_bins, len(samples_in_bin))
            for i in np.random.choice(len(samples_in_bin), n, replace=False): 
                sample_keys.append(samples_in_bin[i])
            
    sample_keys = [sample_keys[i] for i in np.random.permutation(len(sample_keys))]    
    return sample_keys


def load_sampled_frames(sample_keys, video_dir, video_extension=None):
    """
    Load sampled frames from a directory of videos.

    Parameters
    ----------
    sample_keys: list of tuples
        List of sampled frames as tuples with format 
        (key, frame_number, bodypart)

    video_dir: str
        Path to directory containing videos

    video_extension: str, default=None
        Preferred video extension (passed to :py:func:`keypoint_moseq.util.find_matching_videos`)

    Returns
    -------
    sample_keys: dict
        Dictionary mapping elements from ``sample_keys`` to the
        corresponding videos frames.

    """
    keys = sorted(set([k[0] for k in sample_keys]))
    videos = find_matching_videos(keys,video_dir)
    key_to_video = dict(zip(keys,videos))
    readers = {key: OpenCVReader(video) for key,video in zip(keys,videos)}
    pbar = tqdm.tqdm(sample_keys, desc='Loading sample frames', position=0, leave=True)
    return {(key,frame,bodypart):readers[key][frame] for key,frame,bodypart in pbar}


def load_annotations(project_dir):
    """
    Reload saved calibration annotations.

    Parameters
    ----------
    project_dir: str
        Load annotations from ``{project_dir}/error_annotations.csv``

    Returns
    -------
    annotations: dict
        Dictionary mapping sample keys to annotated keypoint 
        coordinates. (See :py:func:`keypoint_moseq.calibration.sample_error_frames` 
        for format of sample keys)
    """   
    annotations = {}
    annotations_path = os.path.join(
        project_dir,'error_annotations.csv')
    if os.path.exists(annotations_path):
        for l in open(annotations_path,'r').read().split('\n')[1:]:
            key,frame,bodypart,x,y = l.split(',')
            sample_key = (key,int(frame),bodypart)
            annotations[sample_key] = (float(x),float(y))
    return annotations
        
def save_annotations(project_dir, annotations): 
    """
    Save calibration annotations to a csv file

    Parameters
    ----------
    project_dir: str
        Save annotations to ``{project_dir}/error_annotations.csv`` 

    annotations: dict
        Dictionary mapping sample keys to annotated keypoint 
        coordinates. (See :py:func:`keypoint_moseq.calibration.sample_error_frames` 
        for format of sample keys)
    """
    output = ['key,frame,bodypart,x,y']
    for (key,frame,bodypart),(x,y) in annotations.items():
        output.append(f'{key},{frame},{bodypart},{x},{y}')
    path = os.path.join(project_dir,'error_annotations.csv')        
    open(path,'w').write('\n'.join(output))
    print(fill(f'Annotations saved to {path}'))
    
def save_params(project_dir, estimator):
    """
    Save config parameters learned via calibration

    Parameters
    ----------
    project_dir: str
        Save parameters ``{project_dir}/config.yml`` 

    estimator: :py:func:`holoviews.streams.Stream`
        Stream object with fields ``conf_threshold``, ``slope``, ``intercept``
    """
    update_config(project_dir, 
                  conf_threshold=float(estimator.conf_threshold),
                  slope=float(estimator.slope), 
                  intercept=float(estimator.intercept))


def _confs_and_dists_from_annotations(coordinates, confidences, 
                                      annotations, bodyparts):
    confs,dists = [],[]
    for (key,frame,bodypart),xy in annotations.items():
        if key in coordinates and key in confidences:
            k = bodyparts.index(bodypart)
            confs.append(confidences[key][frame][k])
            dists.append(np.sqrt(((coordinates[key][frame][k]-np.array(xy))**2).sum()))
    return confs,dists


def _noise_calibration_widget(project_dir, coordinates, confidences,
                              sample_keys, sample_images, annotations, *, 
                              keypoint_colormap, bodyparts, skeleton, 
                              error_estimator, conf_threshold, **kwargs):
    
    from scipy.stats import linregress
    from holoviews.streams import Tap, Stream
    import holoviews as hv
    import panel as pn
    hv.extension('bokeh')

    max_height = np.max([sample_images[k].shape[0] for k in sample_keys]) 
    max_width = np.max([sample_images[k].shape[1] for k in sample_keys]) 
    max_zoom = int(max(max_width,max_height))

    edges = np.array(get_edges(bodyparts,skeleton))
    conf_vals = np.hstack([v.flatten() for v in confidences.values()])
    min_conf,max_conf = np.nanpercentile(conf_vals, .01),np.nanmax(conf_vals)
    
    annotations_stream = Stream.define('Annotations', annotations=annotations)()
    current_sample = Stream.define('Current sample', sample_ix=0)()
    estimator = Stream.define('Estimator', slope=float(error_estimator['slope']),
                              intercept=float(error_estimator['intercept']), 
                              conf_threshold=float(conf_threshold))()
    
    img_tap = Tap(transient=True)
    vline_tap = Tap(transient=True)
    crop_size = hv.Dimension('Crop size', range=(0,max_zoom), default=200)
    
    def update_scatter(x,y, annotations):
        
        confs,dists = _confs_and_dists_from_annotations(
            coordinates, confidences, annotations, bodyparts)
 
        log_dists = np.log10(np.array(dists)+1)
        log_confs = np.log10(np.maximum(confs, min_conf))
        max_dist = np.log10(np.sqrt(max_height**2+max_width**2)+1)

        xspan = np.log10(max_conf)-np.log10(min_conf)
        xlim = (np.log10(min_conf)-xspan/10, np.log10(max_conf)+xspan/10)
        ylim = (-max_dist/50,max_dist)
        
        if len(log_dists)>1:
            m,b = linregress(log_confs,log_dists)[:2]
            estimator.event(slope=m, intercept=b)
        else: m,b = estimator.slope, estimator.intercept

        if x is None: x = np.log10(conf_threshold)
        else: estimator.event(conf_threshold=10**x)
        passing_percent = (conf_vals>10**x).mean()*100
                    
        scatter = hv.Scatter(zip(log_confs,log_dists)).opts(
            color='k', size=6, xlim=xlim, ylim=ylim, axiswise=True, 
            frame_width=250, default_tools=[])
        
        curve = hv.Curve([(xlim[0],xlim[0]*m+b),(xlim[1],xlim[1]*m+b)]).opts(
            xlim=xlim, ylim=ylim, axiswise=True, default_tools=[])  
        
        vline_label = hv.Text(x-(xlim[1]-xlim[0])/50, ylim[1]-(ylim[1]-ylim[0])/100,
            f'confidence\nthreshold\n{10**x:.5f}\n({passing_percent:.1f}%)').opts(
            axiswise=True, text_align='right', text_baseline='top', 
            text_font_size="8pt", default_tools=[])
        
        vline = hv.VLine(x).opts(
            axiswise=True, line_dash='dashed', color='lightgray',  default_tools=[])

        return (scatter*curve*vline*vline_label).opts(
            toolbar=None, default_tools=[], 
            xlabel='log10(confidence)', ylabel='log10(error)')


    def update_img(crop_size, sample_ix, x, y):
        
        key,frame,bodypart = sample_key = sample_keys[sample_ix]
        keypoint_ix = bodyparts.index(bodypart)
        xys = coordinates[key][frame]
        masked_nodes = np.nonzero(~np.isnan(xys).any(1))[0]
        confs = confidences[key][frame]
        
        if x and y:
            annotations_stream.annotations.update({sample_key:(x,y)})
            annotations_stream.event()

        if sample_key in annotations_stream.annotations: 
            point = np.array(annotations_stream.annotations[sample_key])
        else: point = xys[keypoint_ix]
          
        colorvals = np.linspace(0,1,len(bodyparts))
        pt_data = np.append(point,colorvals[keypoint_ix])[None]
        hv_point = hv.Points(pt_data, vdims=['bodypart']).opts(
            color='bodypart', cmap='autumn', size=15, framewise=True, marker='x', line_width=3)
        
        label = f'{bodypart}, confidence = {confs[keypoint_ix]:.5f}'
        h,w = sample_images[sample_key].shape[:2]
        rgb = hv.RGB(sample_images[sample_key][::-1], bounds=(0,0,w,h), label=label).opts(
            framewise=True, xaxis='bare', yaxis='bare', frame_width=250)

        xlim = (xys[keypoint_ix,0]-crop_size/2,xys[keypoint_ix,0]+crop_size/2)
        ylim = (xys[keypoint_ix,1]-crop_size/2,xys[keypoint_ix,1]+crop_size/2)
         
        edge_data = ((),(),())
        if len(edges)>0: 
            masked_edges = edges[np.isin(edges,masked_nodes).all(1)]
            if len(masked_edges)>0:
                edge_data = (*masked_edges.T, colorvals[masked_edges[:,0]])

                    
        sizes = np.where(np.arange(len(xys))==keypoint_ix, 10, 6)[masked_nodes]
        masked_bodyparts = [bodyparts[i] for i in masked_nodes]
        nodes = hv.Nodes((*xys[masked_nodes].T, masked_nodes, masked_bodyparts, sizes), vdims=['name','size'])        
        graph = hv.Graph((edge_data, nodes), vdims='ecolor').opts(
            node_color='name', node_cmap=keypoint_colormap, tools=[],
            edge_color='ecolor', edge_cmap=keypoint_colormap, node_size='size')

        return (rgb*graph*hv_point).opts(data_aspect=1, xlim=xlim, ylim=ylim, toolbar=None)
    
    
    def update_estimator_text(*, slope, intercept, conf_threshold):
        lines = [f'slope: {slope:.6f}',
                 f'intercept: {intercept:.6f}',
                 f'conf_threshold: {conf_threshold:.6f}']
        estimator_textbox.value='<br>'.join(lines)
               
    prev_button = pn.widgets.Button(name='\u25c0', width=50, align='center')
    next_button = pn.widgets.Button(name='\u25b6', width=50, align='center')
    save_button = pn.widgets.Button(name='Save:', width=100, align='center')
    sample_slider = pn.widgets.IntSlider(name='sample', value=0, start=0, end=len(sample_keys), width=100, align='center')
    zoom_slider = pn.widgets.IntSlider(name='Zoom', value=200, start=1, end=max_zoom, width=100, align='center')
    estimator_textbox = pn.widgets.StaticText(align='center')
    
    def next_sample(event):
        if current_sample.sample_ix < len(sample_keys)-1:
            current_sample.event(sample_ix=int(current_sample.sample_ix)+1)
        sample_slider.value=int(current_sample.sample_ix)
        
    def prev_sample(event):
        if current_sample.sample_ix > 0:
            current_sample.event(sample_ix=int(current_sample.sample_ix)-1)
        sample_slider.value=int(current_sample.sample_ix) 
        
    def save_all(event):
        save_annotations(project_dir, annotations_stream.annotations)
        save_params(project_dir, estimator)
    
    @pn.depends(sample_slider.param.value, watch=True)
    def change_sample(value):
        current_sample.event(sample_ix=int(value))
    
    prev_button.on_click(prev_sample)
    next_button.on_click(next_sample)
    save_button.on_click(save_all)
    estimator.add_subscriber(update_estimator_text)
    estimator.event()

    img_dmap = hv.DynamicMap(
        pn.bind(update_img, crop_size=zoom_slider),
        streams=[current_sample, img_tap]
    ).opts(framewise=True)
    
    scatter_dmap = hv.DynamicMap(
        update_scatter, streams=[annotations_stream, vline_tap],
    ).opts(framewise=True, axiswise=True)

    controls = pn.Row(
        prev_button, next_button,
        # sample_slider,
        pn.Spacer(sizing_mode='stretch_width'),
        zoom_slider, 
        pn.Spacer(sizing_mode='stretch_width'),
        save_button,   
        pn.Spacer(sizing_mode='stretch_width'),
        estimator_textbox
    )
    plots = pn.Row(
        img_dmap, 
        scatter_dmap
    )
    return pn.Column(controls, plots)




def noise_calibration(project_dir, coordinates, confidences, *, 
                      bodyparts, use_bodyparts, video_dir, 
                      video_extension=None, conf_pseudocount=0.001, 
                      verbose=False, **kwargs):
    """
    Perform manual annotation to calibrate the relationship between
    keypoint error and neural network confidence. 

    This function creates a widget for interactive annotation in a 
    jupyter notebook. Users mark correct keypoint locations for a 
    sequence of frames, and a regression line is fit to the 
    ``log(confidence), log(error)`` pairs obtained through annotation.
    The regression coefficients are used during modeling to set a 
    prior on the noise level for each keypoint on each frame. 

    Follow these steps to use the widget:
        - After executing this function, a widget should appear with a 
          video frame in the center.
        - Annotate the labeled bodypart in each frame by left-clicking 
          at the correct location. An "X" should appear there.
        - Use the arrow buttons and/or sample slider on the left to 
          annotate additional frames.
        - Each annotation adds a point to the right-hand scatter plot. 
          Continue until the regression line stabilizes.
        - At any point, adjust the confidence threshold by clicking on 
          the scatter plot. The confidence threshold is used to define 
          outlier keypoints for PCA and model initialization.
        - Use the "save" button to store your annotations to disk and 
          save ``slope``, ``intercept``, and ``confidence_threshold``
          to the config.


    Parameters
    ----------
    project_dir: str
        Project directory. Must contain a ``config.yml`` file.

    coordinates: dict
        Keypoint coordinates for a collection of sessions. Values
        must be numpy arrays of shape (T,K,D) where K is the number
        of keypoints and D={2 or 3}. Keys can be any unique str,
        but must start with the name of a videofile in ``video_dir``. 

    confidences: dict
        Nonnegative confidence values for the keypoints in 
        ``coordinates`` as numpy arrays of shape (T,K).

    bodyparts: list
        Label for each keypoint represented in ``coordinates``

    use_bodyparts: list
        Ordered subset of keypoint labels to be used for modeling

    video_dir: str
        Path to directory containing videos. Each video should
        correspond to a key in ``coordinates``. The key must
        contain the videoname as a prefix. 

    video_extension: str, default=None
        Preferred video extension (used in :py:func:`keypoint_moseq.util.find_matching_videos`)

    conf_pseudocount: float, default=0.001
        Pseudocount added to confidence values to avoid log(0) errors.

    verbose: bool, default=False
        Print progress.
    """
    confidences = {k:v+conf_pseudocount for k,v in confidences.items()}
    
    sample_keys = sample_error_frames(
        confidences, bodyparts, use_bodyparts)

    annotations = load_annotations(project_dir)
    sample_keys.extend(annotations.keys())

    sample_images = load_sampled_frames(
        sample_keys, video_dir, video_extension=video_extension)

    return _noise_calibration_widget(
        project_dir, coordinates, confidences, sample_keys, 
        sample_images, annotations, bodyparts=bodyparts, **kwargs)

