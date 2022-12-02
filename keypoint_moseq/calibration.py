import numpy as np
import tqdm
import os
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.io import update_config
from keypoint_moseq.util import find_matching_videos

def sample_error_frames(confidences, bodyparts, use_bodyparts,
                        num_bins=10, num_samples=100, num_videos=10):
    
    all_videos = sorted(confidences.keys())
    num_videos = min(num_videos, len(all_videos))
    use_videos = np.random.choice(all_videos, num_videos, replace=False)
    
    all_confs = np.concatenate([v.flatten() for v in confidences.values()])
    min_conf, max_conf = all_confs.min(), all_confs.max()
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

def load_sample_images(sample_keys, video_dir):
    keys = sorted(set([k[0] for k in sample_keys]))
    videos = find_matching_videos(keys,video_dir)
    key_to_video = dict(zip(keys,videos))
    readers = {key: OpenCVReader(video) for key,video in zip(keys,videos)}
    pbar = tqdm.tqdm(sample_keys, desc='Loading sample frames', position=0, leave=True)
    return {(key,frame,bodypart):readers[key][frame] for key,frame,bodypart in pbar}


def load_annotations(project_dir):
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
    output = ['key,frame,bodypart,x,y']
    for (key,frame,bodypart),(x,y) in annotations.items():
        output.append(f'{key},{frame},{bodypart},{x},{y}')
    path = os.path.join(project_dir,'error_annotations.csv')        
    open(path,'w').write('\n'.join(output))
    print(fill(f'Annotations saved to {path}'))
    
def save_params(project_dir, estimator):
    update_config(project_dir, 
                  conf_threshold=float(estimator.conf_threshold),
                  slope=float(estimator.slope), 
                  intercept=float(estimator.intercept))


def confs_and_dists_from_annotations(coordinates, confidences, 
                                     annotations, bodyparts):
    confs,dists = [],[]
    for (key,frame,bodypart),xy in annotations.items():
        k = bodyparts.index(bodypart)
        confs.append(confidences[key][frame][k])
        dists.append(np.sqrt(((coordinates[key][frame][k]-np.array(xy))**2).sum()))
    return confs,dists


def noise_calibration_widget(project_dir, coordinates, confidences,
                             sample_keys, sample_images, annotations, *, 
                             keypoint_colormap, bodyparts, skeleton, 
                             error_estimator, conf_threshold, 
                             conf_pseudocount, **kwargs):
    
    from scipy.stats import linregress
    from holoviews.streams import Tap, Stream
    import holoviews as hv
    import panel as pn
    hv.extension('bokeh')

    h,w = sample_images[sample_keys[0]].shape[:2] 
    edges = np.array([[bodyparts.index(bp) for bp in edge] for edge in skeleton])
    confidences = {k:v+conf_pseudocount for k,v in confidences.items()}
    conf_vals = np.hstack([v.flatten() for v in confidences.values()])
    min_conf,max_conf = np.percentile(conf_vals, .01),conf_vals.max()
    
    annotations_stream = Stream.define('Annotations', annotations=annotations)()
    current_sample = Stream.define('Current sample', sample_ix=0)()
    estimator = Stream.define('Estimator', slope=float(error_estimator['slope']),
                              intercept=float(error_estimator['intercept']), 
                              conf_threshold=float(conf_threshold))()
    
    img_tap = Tap(transient=True)
    vline_tap = Tap(transient=True)
    crop_size = hv.Dimension('Crop size', range=(0,max(w,h)), default=200)
    img = hv.RGB(sample_images[sample_keys[0]], bounds=(0,0,w,h))
        
    
    def update_scatter(x,y, annotations):
        
        confs,dists = confs_and_dists_from_annotations(
            coordinates, confidences, annotations, bodyparts)
        
        log_dists = np.log10(np.array(dists)+1)
        log_confs = np.log10(np.maximum(confs, min_conf))
        max_dist = np.log10(np.sqrt(h**2+w**2)+1)

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
            color='k', size=6, xlim=xlim, ylim=ylim, axiswise=True, default_tools=[])
        
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
        confs = confidences[key][frame]
        
        if x and y:
            annotations_stream.annotations.update({sample_key:(x,y)})
            annotations_stream.event()
        else: img.data = sample_images[sample_key][::-1] 

        if sample_key in annotations_stream.annotations: 
            point = np.array(annotations_stream.annotations[sample_key])
        else: point = xys[keypoint_ix]
          
        colorvals = np.linspace(0,1,len(bodyparts))
        pt_data = np.append(point,colorvals[keypoint_ix])[None]
        hv_point = hv.Points(pt_data, vdims=['bodypart']).opts(
            color='bodypart', cmap='autumn', size=15, framewise=True, marker='x', line_width=3)
        
        sizes = np.where(np.arange(len(xys))==keypoint_ix, 10, 6)
        nodes = hv.Nodes((*xys.T, np.arange(len(bodyparts)), bodyparts, sizes), vdims=['name','size'])
        graph = hv.Graph(((*edges.T, colorvals[edges[:,0]]), nodes), vdims='ecolor').opts(
            node_color='name', node_cmap=keypoint_colormap, tools=[],
            edge_color='ecolor', edge_cmap=keypoint_colormap, node_size='size')

        label = f'{bodypart}, confidence = {confs[keypoint_ix]:.5f}'
        rgb = hv.RGB(img, bounds=(0,0,w,h), label=label).opts(framewise=True)

        xlim = (xys[keypoint_ix,0]-crop_size/2,xys[keypoint_ix,0]+crop_size/2)
        ylim = (xys[keypoint_ix,1]-crop_size/2,xys[keypoint_ix,1]+crop_size/2)
        return (rgb*graph*hv_point).opts(data_aspect=1, xlim=xlim, ylim=ylim, toolbar=None)
    
    
    def update_estimator_text(*, slope, intercept, conf_threshold):
        lines = [f'slope: {slope:.6f}',
                 f'intercept: {intercept:.6f}',
                 f'conf_threshold: {conf_threshold:.6f}']
        estimator_textbox.value='<br>'.join(lines)
    

               
    prev_button = pn.widgets.Button(name='\u25c0', width=50, height=30)
    next_button = pn.widgets.Button(name='\u25b6', width=50, height=30)
    save_button = pn.widgets.Button(name='Save', width=100, align='center')
    sample_slider = pn.widgets.IntSlider(name='sample', value=0, start=0, end=len(sample_keys), width=120, align='center')
    zoom_slider = pn.widgets.IntSlider(name='Zoom', value=200, start=1, end=max(w,h), width=120, align='center')
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
                    
    controls = pn.Column(
        pn.Row(prev_button, next_button, align='center'), 
        sample_slider,
        zoom_slider,
        pn.layout.Divider(),
        estimator_textbox,
        pn.layout.Divider(),
        save_button,
        width=170)

    return pn.Row(controls, img_dmap, scatter_dmap)


def noise_calibration(project_dir, coordinates, confidences, *, 
                      bodyparts, use_bodyparts, video_dir, **kwargs):

    sample_keys = sample_error_frames(confidences, bodyparts, use_bodyparts)
    annotations = load_annotations(project_dir)
    sample_keys.extend(annotations.keys())
    sample_images = load_sample_images(sample_keys, video_dir)

    return noise_calibration_widget(
        project_dir, coordinates, confidences, sample_keys, 
        sample_images, annotations, bodyparts=bodyparts, **kwargs)

