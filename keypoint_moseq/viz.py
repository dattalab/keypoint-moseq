import os
import cv2
import tqdm
import imageio
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from vidio.read import OpenCVReader
from textwrap import fill

from keypoint_moseq.util import (
    get_edges, get_durations, get_frequencies, reindex_by_bodyparts,
    find_matching_videos, get_syllable_instances, sample_instances,
    filter_centroids_headings, get_trajectories, interpolate_keypoints,
    interpolate_along_axis
)
from keypoint_moseq.io import load_results
from jax_moseq.models.keypoint_slds import center_embedding

# simple warning formatting
plt.rcParams['figure.dpi'] = 100
warnings.formatwarning = lambda msg, *a: str(msg)

# suppress warnings from imageio
logging.getLogger().setLevel(logging.ERROR)



def crop_image(image, centroid, crop_size):
    """
    Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int or tuple(int,int)
        Size of the crop around the centroid. Either a single int for
        a square crop, or a tuple of ints (w,h) for a rectangular crop.


    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    if isinstance(crop_size,tuple): w,h = crop_size
    else: w,h = crop_size,crop_size
    x,y = int(centroid[0]),int(centroid[1])

    x_min = max(0, x - w//2)
    y_min = max(0, y - h//2)
    x_max = min(image.shape[1], x + w//2)
    y_max = min(image.shape[0], y + h//2)

    cropped = image[y_min:y_max, x_min:x_max]
    padded = np.zeros((h,w,*image.shape[2:]), dtype=image.dtype)
    pad_x = (w - cropped.shape[1]) // 2
    pad_y = (h - cropped.shape[0]) // 2
    padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
    return padded


def plot_scree(pca, savefig=True, project_dir=None, fig_size=(3,2),
              ):
    """
    Plot explained variance as a function of the number of PCs.

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is 
        saved to ``{project_dir}/pca_scree.pdf``.

    project_dir : str, default=None
        Path to the project directory. Required if ``savefig`` is True.

    fig_size : tuple, (2.5,2)
        Size of the figure in inches.

    Returns 
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Figure handle
    """
    fig = plt.figure()
    plt.plot(np.arange(len(pca.mean_))+1,np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('PCs')
    plt.ylabel('Explained variance')
    plt.gcf().set_size_inches(fig_size)
    plt.grid()
    plt.tight_layout()
    
    if savefig:
        assert project_dir is not None, fill(
            'The ``savefig`` option requires a ``project_dir``')
        plt.savefig(os.path.join(project_dir,'pca_scree.pdf'))
    plt.show()
    return fig
          
def plot_pcs(pca, *, use_bodyparts, skeleton, keypoint_colormap='autumn',
             savefig=True, project_dir=None, scale=1, plot_n_pcs=10, 
             axis_size=(2,1.5), ncols=5, node_size=30.0, linewidth=2.0, **kwargs):
    """
    Visualize the components of a fitted PCA model.

    For each PC, a subplot shows the mean pose (semi-transparent) along
    with a perturbation of the mean pose in the direction of the PC. 

    Parameters
    ----------
    pca : :py:func:`sklearn.decomposition.PCA`
        Fitted PCA model

    use_bodyparts : list of str
        List of bodyparts to that are used in the model; used to index
        bodypart names in the skeleton.

    skeleton : list
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names.

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    savefig : bool, True
        Whether to save the figure to a file. If true, the figure is
        saved to ``{project_dir}/pcs-{xy/xz/yz}.pdf`` (``xz`` and ``yz``
        are only included for 3D data).

    project_dir : str, default=None
        Path to the project directory. Required if ``savefig`` is True.

    scale : float, default=0.5
        Scale factor for the perturbation of the mean pose.

    plot_n_pcs : int, default=10
        Number of PCs to plot. 

    axis_size : tuple of float, default=(2,1.5)
        Size of each subplot in inches.

    ncols : int, default=5
        Number of columns in the figure.

    node_size : float, default=30.0
        Size of the keypoints in the figure.
        
    linewidth: float, default=2.0
        Width of edges in skeleton
    """
    k = len(use_bodyparts)
    d = len(pca.mean_)//(k-1)  
    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    cmap = plt.cm.get_cmap(keypoint_colormap)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])
    
    if d==2: dims_list,names = [[0,1]],['xy']
    if d==3: dims_list,names = [[0,1],[0,2]],['xy','xz']
    
    magnitude = np.sqrt((pca.mean_**2).mean()) * scale
    for dims,name in zip(dims_list,names):
        nrows = int(np.ceil(plot_n_pcs/ncols))
        fig,axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i,ax in enumerate(axs.flat):
            ymean = Gamma @ pca.mean_.reshape(k-1,d)[:,dims]
            y = Gamma @ (pca.mean_ + magnitude*pca.components_[i]).reshape(k-1,d)[:,dims]
            
            for e in edges:  
                ax.plot(*ymean[e].T, color=cmap(e[0]/(k-1)), 
                        zorder=0, alpha=0.25, linewidth=linewidth)
                ax.plot(*y[e].T, color='k', 
                        zorder=2, linewidth=linewidth+.2)
                ax.plot(*y[e].T, color=cmap(e[0]/(k-1)), 
                        zorder=3, linewidth=linewidth)
                
            ax.scatter(*ymean.T, c=np.arange(k), cmap=cmap, s=node_size, 
                       zorder=1, alpha=0.25, linewidth=0)
            ax.scatter(*y.T, c=np.arange(k), cmap=cmap, s=node_size, 
                       zorder=4, edgecolor='k', linewidth=0.2)
            
            ax.set_title(f'PC {i+1}', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.set_size_inches((axis_size[0]*ncols, axis_size[1]*nrows))
        plt.tight_layout()
        
        if savefig:
            assert project_dir is not None, fill(
                'The ``savefig`` option requires a ``project_dir``')
            plt.savefig(os.path.join(project_dir,f'pcs-{name}.pdf'))
        plt.show()
        

def plot_syllable_frequencies(results=None, path=None, project_dir=None, 
                              name=None, use_reindexed=True, minlength=10,
                              min_frequency=0.005):
    """
    Plot a histogram showing the frequency of each syllable.
    
    Caller must provide a results dictionary, a path to a results .h5,
    or a project directory and model name, in which case the results are
    loaded from ``{project_dir}/{name}/results.h5``.

    Parameters
    ----------
    results : dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.apply_model`)

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``path`` is None. 
        
    project_dir: str, default=None
        Project directory. Required to load results if ``results`` is 
        None and ``path`` is None. 

    path: str, default=None
        Path to a results file. If None, results will be loaded from
        ``{project_dir}/{name}/results.h5``.

    use_reindexed: bool, default=True
        Whether to use label syllables by their frequency rank (True) or
        or their original label (False). When reindexing, "0"  represents
        the most frequent syllable).

    minlength: int, default=10
        Minimum x-axis length of the histogram.

    min_frequency: float, default=0.005
        Minimum frequency of syllables to include in the histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the histogram.
    
    ax : matplotlib.axes.Axes
        Axes containing the histogram.
    """
    if results is None:
        results = load_results(path=path, name=name, project_dir=project_dir)

    syllable_key = 'syllables' if not use_reindexed else 'syllables_reindexed'
    syllables = {k:res[syllable_key] for k,res in results.items()}
    frequencies = get_frequencies(syllables)
    frequencies = frequencies[frequencies>min_frequency]
    xmax = max(minlength, len(frequencies))

    fig, ax = plt.subplots()
    ax.bar(range(len(frequencies)),frequencies,width=1)
    ax.set_ylabel('probability')
    ax.set_xlabel('syllable rank')
    ax.set_xlim(-1,xmax+1)
    ax.set_title('Frequency distribution')
    ax.set_yticks([])
    return fig, ax


def plot_duration_distribution(results=None, path=None, project_dir=None, 
                               name=None, use_reindexed=True, lim=None,
                               num_bins=30, fps=None):
    """
    Plot a histogram showing the frequency of each syllable.
    
    Caller must provide a results dictionary, a path to a results .h5,
    or a project directory and model name, in which case the results are
    loaded from ``{project_dir}/{name}/results.h5``.

    Parameters
    ----------
    results : dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.apply_model`)

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``path`` is None. 
        
    project_dir: str, default=None
        Project directory. Required to load results if ``results`` is 
        None and ``path`` is None. 

    path: str, default=None
        Path to a results file. If None, results will be loaded from
        ``{project_dir}/{name}/results.h5``.

    lim: tuple, default=None
        x-axis limits as a pair of ints (in units of frames). If None,
        the limits are set to (0, 95th-percentile).

    num_bins: int, default=30
        Number of bins in the histogram.

    fps: int, default=None
        Frames per second. Used to convert x-axis from frames to seconds.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the histogram.
    
    ax : matplotlib.axes.Axes
        Axes containing the histogram.
    """
    if results is None:
        results = load_results(path=path, name=name, project_dir=project_dir)

    syllable_key = 'syllables' if not use_reindexed else 'syllables_reindexed'
    syllables = {k:res[syllable_key] for k,res in results.items()}
    durations = get_durations(syllables)
    
    if lim is None:
        lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim/num_bins)),1)

    if fps is not None:
        durations = durations/fps
        binsize = binsize/fps
        lim = lim/fps
        xlabel = 'syllable duration (s)'
    else:
        xlabel = 'syllable duration (frames)'

    fig, ax = plt.subplots()
    ax.hist(durations, range=(0,lim), bins=(int(lim/binsize)), density=True)
    ax.set_xlim([0,lim])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability')
    ax.set_title('Duration distribution')
    ax.set_yticks([])
    return fig, ax
        

def plot_progress(model, data, history, iteration, path=None,
                  project_dir=None, name=None, savefig=True,
                  fig_size=None, seq_length=600, min_frequency=.001, 
                  min_histogram_length=10, **kwargs):
    """
    Plot the progress of the model during fitting.

    The figure shows the following plots:
        - Duration distribution: 
            The distribution of state durations for the most recent
            iteration of the model.
        - Frequency distribution:
            The distribution of state frequencies for the most recent
            iteration of the model.
        - Median duration:
            The median state duration across iterations.
        - State sequence history
            The state sequence across iterations in a random window 
            (a new window is selected each time the progress is plotted). 

    Parameters
    ----------
    model : dict
        Model dictionary containing ``states``

    data : dict
        Data dictionary containing ``mask``

    history : dict
        Dictionary mapping iteration number to saved model dictionaries

    iteration : int
        Current iteration of model fitting

    savefig : bool, default=True
        Whether to save the figure to a file. If true, the figure is
        either saved to ``path`` or, to ``{project_dir}/{name}-progress.pdf``
        if ``path`` is None.

    fig_size : tuple of float, default=None
        Size of the figure in inches. 
        
    seq_length : int, default=600
        Length of the state sequence history plot.

    min_frequency : float, default=.001
        Minimum frequency for including a state in the frequency 
        distribution plot.

    min_histogram_length : int, default=10
        Minimum x-axis length of the frequency distribution plot.

    project_dir : str, default=None
    name : str, default=None
    path : str, default=None

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plots.

    axs : list of matplotlib.axes.Axes
        Axes containing the plots.
    """
    z = np.array(model['states']['z'])
    mask = np.array(data['mask'])
    durations = get_durations(z,mask)
    frequencies = get_frequencies(z,mask)
    
    history_iters = np.array(sorted(history.keys()))
    past_stateseqs = [history[i]['states']['z'] 
                      for i in history_iters 
                      if 'states' in history[i]]
        
    if len(past_stateseqs)>0: 
        fig,axs = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,3]})
        if fig_size is None: fig_size=(12,2.5)
    else: 
        fig,axs = plt.subplots(1,2)
        if fig_size is None: fig_size=(4,2.5)

    frequencies = np.sort(frequencies[frequencies>min_frequency])[::-1]
    xmax = max(len(frequencies),min_histogram_length)
    axs[0].bar(range(len(frequencies)),frequencies,width=1)
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('syllable rank')
    axs[0].set_xlim([-1,xmax+1])
    axs[0].set_title('Frequency distribution')
    axs[0].set_yticks([])
    
    lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim/30)),1)
    axs[1].hist(durations, range=(1,lim), bins=(int(lim/binsize)), density=True)
    axs[1].set_xlim([1,lim])
    axs[1].set_xlabel('syllable duration (frames)')
    axs[1].set_ylabel('probability')
    axs[1].set_title('Duration distribution')
    axs[1].set_yticks([])
    
    if len(past_stateseqs)>0:
        
        med_durs = [np.median(get_durations(z,mask)) for z in past_stateseqs]
        axs[2].scatter(history_iters,med_durs)
        axs[2].set_ylim([-1,np.max(med_durs)*1.1])
        axs[2].set_xlabel('iteration')
        axs[2].set_ylabel('duration')
        axs[2].set_title('Median duration')
        
        nz = np.stack(np.array(mask[:,seq_length:]).nonzero(),axis=1)
        batch_ix,start = nz[np.random.randint(nz.shape[0])]
        seq_hist = np.stack([z[batch_ix,start:start+seq_length] for z in past_stateseqs])
        axs[3].imshow(seq_hist, cmap=plt.cm.jet, aspect='auto', interpolation='nearest')
        axs[3].set_xlabel('Time (frames)')
        axs[3].set_ylabel('Iterations')
        axs[3].set_title('State sequence history')
        
        yticks = [int(y) for y in axs[3].get_yticks() if y < len(history_iters) and y > 0]
        yticklabels = history_iters[yticks]
        axs[3].set_yticks(yticks)
        axs[3].set_yticklabels(yticklabels)

    title = f'Iteration {iteration}'
    if name is not None: title = f'{name}: {title}'
    fig.suptitle(title)        
    fig.set_size_inches(fig_size)
    plt.tight_layout()
    
    if savefig:
        if path is None:
            assert name is not None and project_dir is not None, fill(
                'The ``savefig`` option requires either a ``path`` '
                'or a ``name`` and ``project_dir``')
            path = os.path.join(project_dir,name,'fitting_progress.pdf')
        plt.savefig(path)  
    plt.show()

    return fig,axs

    
def write_video_clip(frames, path, fps=30, quality=7):
    """
    Write a video clip to a file.

    Parameters
    ----------
    frames : np.ndarray
        Video frames as a 4D array of shape `(num_frames, height, width, 3)`
        or a 3D array of shape `(num_frames, height, width)`.

    path : str
        Path to save the video clip.

    fps : int, default=30
        Framerate of video encoding.

    quality : int, default=7
        Quality of video encoding.
    """
    with imageio.get_writer(
        path, pixelformat='yuv420p', 
        fps=fps, quality=quality) as writer:
        for frame in frames: 
            writer.append_data(frame)


def _grid_movie_tile(key, start, end, videos, centroids, headings, 
                     dot_color=(255,255,255), window_size=112,
                     pre=30, post=60, dot_radius=4):
            
        cs = centroids[key][start-pre:start+post]
        h,c = headings[key][start],cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)],[-np.sin(h), np.cos(h)]])
        c = r @ c - window_size//2
        M = [[ np.cos(h), np.sin(h),-c[0]], [-np.sin(h), np.cos(h),-c[1]]]
        
        tile = []
        frames = videos[key][start-pre:start+post]
        for ii,(frame,c) in enumerate(zip(frames,cs)):
            frame = cv2.warpAffine(frame,np.float32(M),(window_size,window_size))
            if 0 <= ii-pre <= end-start and dot_radius>0:
                pos = tuple([int(x) for x in M@np.append(c,1)])
                cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            tile.append(frame)  
        return np.stack(tile)


    
def grid_movie(instances, rows, cols, videos, centroids, headings,
               dot_color=(255,255,255), dot_radius=4, window_size=112, 
               pre=30, post=60, coordinates=None, overlay_trajectory=False,
               plot_options={}, overlay_options={}):
    
    """Generate a grid movie and return it as an array of frames.

    Grid movies show many instances of a syllable. Each instance
    contains a snippet of a video, centered on the animal and synchronized
    to the onset of the syllable. A dot appears at syllable onset and 
    disappears at syllable offset.

    Parameters
    ----------
    instances: list of tuples ``(key, start, end)``
        List of syllable instances to include in the grid movie,
        where each instance is specified as a tuple with the video 
        name, start frame and end frame. The list must have length
        ``rows*cols``. The video names must also be keys in ``videos``.
        
    rows: int, cols : int
        Number of rows and columns in the grid movie grid
    
    videos: dict
        Dictionary mapping video names to video readers. Frames from
        each reader should be accessible via ``__getitem__(int or slice)``

    centroids: dict
        Dictionary mapping video names to arrays of shape ``(n_frames, 2)``
        with the x,y coordinates of animal centroid on each frame

    headings: dict
        Dictionary mapping video names to arrays of shape ``(n_frames,)``
        with the heading of the animal on each frame (in radians)

    dot_color: tuple of ints, default=(255,255,255)
        RGB color of the dot indicating syllable onset and offset

    dot_radius: int, default=4
        Radius of the dot indicating syllable onset and offset

    window_size: int, default=112
        Size of the window around the animal. This should be a multiple
        of 16 or imageio will complain.

    pre: int, default=30
        Number of frames before syllable onset to include in the movie

    post: int, default=60
        Number of frames after syllable onset to include in the movie

    coordinates: dict, default=None
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). Required when
        ``overlay_trajectory=True``

    overlay_trajectory: bool, default=False
        Whether to overlay trajectory of keypoints on the grid movie. 
        If True, ``coordinates`` must be provided.

    overlay_options: dict, default={}
        Dictionary of options for overlaying trajectory (see
        :py:func:`keypoint_moseq.viz.overlay_trajectory_on_video`).

    plot_options: dict, default={}
        Dictionary of options for overlaying trajectory (see 
        :py:func:`keypoint_moseq.viz.overlay_keypoints_on_image`).

    Returns
    -------
    frames: array of shape ``(rows, cols, post+pre, window_size, window_size, 3)``
        Array of frames in the grid movie
    """
    if overlay_trajectory:
        assert coordinates is not None, fill(
            '``coordinates`` must be provided if ``overlay_trajectory`` is True')

        trajectories = get_trajectories(
            instances, coordinates, pre=pre, post=post, 
            centroids=centroids, headings=headings)
            
    tiles = []
    for i,(key,start,end) in enumerate(instances):
        tile = _grid_movie_tile(
            key, start, end, videos, centroids, headings, 
            dot_color=dot_color, window_size=window_size,
            pre=pre, post=post, dot_radius=dot_radius)
        
        if overlay_trajectory:
            trajectory = trajectories[i] + window_size//2
            tile = overlay_trajectory_on_video(
                tile, trajectory, pre, plot_options=plot_options, **overlay_options)
        tiles.append(tile)

    tiles = np.stack(tiles).reshape(rows, cols, post+pre, window_size, window_size, 3)
    frames = np.concatenate(np.concatenate(tiles,axis=2),axis=2)
    return frames


def generate_grid_movies(
    results=None, output_dir=None, name=None, project_dir=None,
    results_path=None, video_dir=None, video_paths=None, rows=4, 
    cols=6, filter_size=9, pre=30, post=60, min_frequency=0.005, 
    min_duration=3, dot_radius=4, dot_color=(255,255,255), 
    window_size=112, use_reindexed=True, coordinates=None, 
    bodyparts=None, use_bodyparts=None, skeleton=[], quality=7, 
    sampling_options={},  overlay_trajectory=False, plot_options={},
    overlay_options={}, video_extension=None, **kwargs):
    
    """
    Generate grid movies for a modeled dataset.

    Grid movies show many instances of a syllable and are useful in
    figuring out what behavior the syllable captures 
    (see :py:func:`keypoint_moseq.viz.grid_movie`). This method
    generates a grid movie for each syllable that is used sufficiently
    often (i.e. has at least ``rows*cols`` instances with duration
    of at least ``min_duration`` and an overall frequency of at least
    ``min_frequency``). The grid movies are saved to ``output_dir`` if 
    specified, or else to ``{project_dir}/{name}/grid_movies``.

    Parameters
    ----------
    results: dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.apply_model`). Must have
        the format::

            {
                session_name1: {
                    'syllables':              array of shape (n_frames,),
                    'syllables_reindexed':    array of shape (n_frames,),
                    'centroid':               array of shape (n_frames, dim),
                    'heading' :               array of shape (n_frames,), 
                },
                ...  
            }
            
        - ``syllables`` is required if ``use_reindexed=False``
        - ``syllables_reindexed`` is required if ``use_reindexed=True``
        - ``centroid`` is always required
        - ``heading`` is always required

        If ``results=None``, results will be loaded using either 
        ``results_path`` or  ``project_dir`` and ``name``.

    output_dir: str, default=None
        Directory where grid movies should be saved. If None, grid
        movies will be saved to ``{project_dir}/{name}/grid_movies``.

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save grid movies 
        if ``output_dir`` is None.
        
    project_dir: str, default=None
        Project directory. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save grid movies 
        if ``output_dir`` is None.

    results_path: str, default=None
        Path to a results file. If None, results will be loaded from
        ``{project_dir}/{name}/results.h5``.

    video_dir: str, default=None
        Directory containing videos of the modeled data (see 
        :py:func:`keypoint_moseq.io.find_matching_videos`). If None,
        a dictionary of ``video_paths`` must be provided.

    video_paths: dict, default=None
        Dictionary mapping session names to video paths. The session 
        names must correspond to keys in ``results['syllables']``. If
        None, a ``video_dir`` must be provided.

    filter_size: int, default=9
        Size of the median filter applied to centroids and headings

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to be included in the grid movies.

    min_duration: int, default=3
        Minimum duration of a syllable instance to be included in the 
        grid movie for that syllable. 

    use_reindexed: bool, default=True
        Whether to use label syllables by their frequency rank (True) or
        or their original label (False). When reindexing, "0"  represents
        the most frequent syllable).

    sampling_options: dict, default={}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).
    
    coordinates: dict, default=None
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). Required when
        ``overlay_trajectory=True``, and for density-based sampling 
        (i.e. when ``sampling_options['mode']=='density'``; see 
        :py:func:`keypoint_moseq.util.sample_instances`).

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. Required when 
        ``coordinates`` is provided and bodyparts were reindexed 
        for modeling. 

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts used for modeling. Required when 
        ``coordinates`` is provided and bodyparts were reindexed 
        for modeling. 

    skeleton : list, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.

    overlay_trajectory: bool, default=False
        Whether to overlay trajectory of keypoints on the grid movie. 
        If True, ``coordinates`` must be provided.

    overlay_options: dict, default={}
        Dictionary of options for overlaying trajectory (see
        :py:func:`keypoint_moseq.viz.overlay_trajectory_on_video`).

    plot_options: dict, default={}
        Dictionary of options for overlaying trajectory (see 
        :py:func:`keypoint_moseq.viz.overlay_keypoints_on_image`).

    quality: int, default=7
        Quality of the grid movies. Higher values result in higher
        quality movies but larger file sizes.

    rows, cols, pre, post, dot_radius, dot_color, window_size
        See :py:func:`keypoint_moseq.viz.grid_movie`

    video_extension: str, default=None
        Preferred video extension (passed to :py:func:`keypoint_moseq.util.find_matching_videos`)
    """
    assert (video_dir is not None) or (video_paths is not None), fill(
        'You must provide either ``video_dir`` or ``video_paths``') 

    if overlay_trajectory:
        assert coordinates is not None, fill(
            '``coordinates`` must be provided if ``overlay_trajectory`` is True')     
    
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where grid movies should '
            'be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'grid_movies')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Writing grid movies to {output_dir}')
    
    if not (bodyparts is None or use_bodyparts is None or coordinates is None):
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    if results is None: results = load_results(
        name=name, project_dir=project_dir, path=results_path)
    
    if video_paths is None:
        video_paths = find_matching_videos(
            results.keys(), video_dir, as_dict=True, 
            video_extension=video_extension)

    videos = {k: OpenCVReader(path) for k,path in video_paths.items()}
    fps = list(videos.values())[0].fps
    
    syllable_key = 'syllables' + ('_reindexed' if use_reindexed else '')
    syllables = {k:v[syllable_key] for k,v in results.items()}
    centroids = {k:v['centroid'] for k,v in results.items()}
    headings = {k:v['heading'] for k,v in results.items()}

    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=rows*cols)
    
    if len(syllable_instances) == 0:
        warnings.warn(fill(
            'No syllables with sufficient instances to make a grid movie. '
            'This usually occurs when all frames have the same syllable label '
            '(use `plot_syllable_frequencies` to check if this is the case)'))
        return

    sampled_instances = sample_instances(
        syllable_instances, rows*cols, coordinates=coordinates, 
        centroids=centroids, headings=headings, **sampling_options)

    centroids,headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size)
    
    if len(skeleton)>0 and overlay_trajectory: 
        if isinstance(skeleton[0][0],str):
            assert use_bodyparts is not None, fill(
                'If skeleton edges are specified using bodypart names, '
                '``use_bodyparts`` must be specified')
            plot_options['edges'] = get_edges(use_bodyparts, skeleton)
        else: plot_options['edges'] = skeleton
    
    for syllable,instances in tqdm.tqdm(
        sampled_instances.items(), desc='Generating grid movies'):
        
        frames = grid_movie(
            instances, rows, cols, videos, centroids, headings,
            window_size=window_size, dot_color=dot_color, pre=pre, post=post,
            dot_radius=dot_radius, overlay_trajectory=overlay_trajectory,
            coordinates=coordinates, plot_options=plot_options,
            overlay_options=overlay_options)

        path = os.path.join(output_dir, f'syllable{syllable}.mp4')
        write_video_clip(frames, path, fps=fps, quality=quality)

        
def get_limits(coordinates, pctl=1, blocksize=None,
               left=0.2, right=0.2, top=0.2, bottom=0.2):
    """
    Get axis limits based on the coordinates of all keypoints.

    For each axis, limits are determined using the percentiles
    ``pctl`` and ``100-pctl`` and then padded by ``padding``.

    Parameters
    ----------
    coordinates: ndarray or dict
        Coordinates as an ndarray of shape (..., 2), or a dict
        with values that are ndarrays of shape (..., 2).

    pctl: float, default=1
        Percentile to use for determining the axis limits.

    blocksize: int, default=None
        Axis limits are cast to integers and padded so that the width
        and height are multiples of ``blocksize``. This is useful
        when they are used for generating cropped images for a video. 
        
    left, right, top, bottom: float, default=0.1
        Fraction of the axis range to pad on each side.

    Returns
    -------
    lims: ndarray of shape (2,dim)
        Axis limits, in the format ``[[xmin,ymin,...],[xmax,ymax,...]]``.
    """
    if isinstance(coordinates, dict):
        X = np.concatenate(list(coordinates.values())).reshape(-1,2)
    else:
        X = coordinates.reshape(-1,2)

    xmin,ymin = np.nanpercentile(X, pctl, axis=0)
    xmax,ymax = np.nanpercentile(X, 100-pctl, axis=0)

    width = xmax-xmin
    height = ymax-ymin
    xmin -= width*left
    xmax += width*right
    ymin -= height*bottom
    ymax += height*top

    lims = np.array([
        [xmin,ymin],
        [xmax,ymax]])

    if blocksize is not None:
        lims = np.round(lims)
        padding = np.mod(lims[0]-lims[1], blocksize)/2
        lims[0] -= padding
        lims[1] += padding
        lims = lims.astype(int)
    
    return lims

def rasterize_figure(fig):
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    raster_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    raster = raster_flat.reshape((height, width, 3))
    return raster


def plot_trajectories(titles, Xs, lims, edges=[], n_cols=4, invert=False, 
                      keypoint_colormap='autumn', node_size=50, line_width=3, 
                      alpha=0.2, num_timesteps=10, plot_width=4, overlap=(0.2,0),
                      return_rasters=False):
    """
    Plot one or more pose trajectories on a common axis and return
    the axis.

    (See :py:func:`keypoint_moseq.viz.generate_trajectory_plots`)

    Parameters
    ----------
    titles: list of str
        List of titles for each trajectory plot.

    Xs: list of ndarray
        List of pose trajectories as ndarrays of shape
        (n_frames, n_keypoints, 2).

    edges: list of tuples, default=[]
        List of edges, where each edge is a tuple of two integers

    lims: ndarray
        Axis limits used for all the trajectory plots. The limits 
        should be provided as an array of shape (2,2) with the format
        ``[[xmin,ymin],[xmax,ymax]]``.

    n_cols: int, default=4
        Number of columns in the figure (used when plotting multiple
        trajectories).

    invert: bool, default=False
        Determines the background color of the figure. If ``True``,
        the background will be black.

    keypoint_colormap : str or list
        Name of a matplotlib colormap or a list of colors as (r,b,g) 
        tuples in the same order as as the keypoints.

    node_size: int, default=50
        Size of each keypoint.

    line_width: int, default=3
        Width of the lines connecting keypoints.

    alpha: float, default=0.2
        Opacity of fade-out layers.

    num_timesteps: int, default=10
        Number of timesteps to plot for each trajectory. The pose 
        at each timestep is determined by linearly interpolating
        between the keypoints.

    plot_width: int, default=4
        Width of each trajectory plot in inches. The height  is 
        determined by the aspect ratio of ``lims``. The final figure 
        width is ``fig_width * min(n_cols, len(X))``.

    overlap: tuple of float, default=(0.2,0)
        Amount of overlap between each trajectory plot as a tuple 
        with the format ``(x_overlap, y_overlap)``. The values should
        be between 0 and 1.
        
    return_rasters: bool, default=False
        Rasterize the matplotlib canvas after plotting each step of
        the trajecory. This is used to generate an animated video/gif
        of the trajectory. 

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Figure handle

    ax: matplotlib.axes.Axes
        Axis containing the trajectory plots.
    """
    fill_color = 'k' if invert else 'w'
    if isinstance(keypoint_colormap, list): colors = keypoint_colormap
    else: colors = plt.cm.get_cmap(keypoint_colormap)(np.linspace(0,1,Xs[0].shape[1]))

    n_cols = min(n_cols, len(Xs))
    n_rows = np.ceil(len(Xs)/n_cols)
    offsets = np.stack(np.meshgrid(
        np.arange(n_cols)*np.diff(lims[:,0])*(1-overlap[0]),
        np.arange(n_rows)*np.diff(lims[:,1])*(overlap[1]-1)
    ),axis=-1).reshape(-1,2)[:len(Xs)]
    
    Xs = interpolate_along_axis(
        np.linspace(0,Xs[0].shape[0],num_timesteps), 
        np.arange(Xs[0].shape[0]), np.array(Xs), axis=1)

    Xs = Xs+offsets[:,None,None]
    xmin,ymin = lims[0] + offsets.min(0)
    xmax,ymax = lims[1] + offsets.max(0)
    
    fig,ax = plt.subplots(frameon=False)
    ax.fill_between(
        [xmin,xmax], y1=[ymax,ymax], y2=[ymin,ymin], 
        facecolor=fill_color, zorder=0, clip_on=False)
    
    title_xy = (lims * np.array([[0.5,0.1],[0.5,0.9]])).sum(0)
    title_color = 'w' if invert else 'k'

    for xy,text in zip(offsets+title_xy,titles):
        ax.text(*xy, text, c=title_color, ha='center', 
                va='top', zorder=Xs.shape[1]*4+4)
        
    # final extents in axis
    final_width = xmax-xmin
    final_height = title_xy[1]-ymin
    
    fig_width = plot_width*(n_cols - (n_cols-1)*overlap[0])
    fig_height = final_height/final_width*fig_width
    fig.set_size_inches((fig_width, fig_height))
        
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
        
    rasters = [] # for making a gif
    
    for i in range(Xs.shape[1]):
        for X,offset in zip(Xs,offsets):
            for ii,jj in edges: 
                ax.plot(*X[i,(ii,jj)].T, c='k', zorder=i*4, 
                        linewidth=line_width, clip_on=False)
        
            for ii,jj in edges: 
                ax.plot(*X[i,(ii,jj)].T, c=colors[ii], zorder=i*4+1, 
                        linewidth=line_width*.9, clip_on=False)

            ax.scatter(*X[i].T, c=colors, zorder=i*4+2, edgecolor='k', 
                       linewidth=0.4, s=node_size, clip_on=False)
        
        if i < Xs.shape[1]-1: 
            ax.fill_between(
                [xmin,xmax], y1=[ymax,ymax], y2=[ymin,ymin], 
                facecolor=fill_color, alpha=alpha, zorder=i*4+3, clip_on=False)
 
        if return_rasters:
            rasters.append(rasterize_figure(fig))          

    return fig,ax,rasters

def generate_trajectory_plots(
    coordinates=None, results=None, output_dir=None, name=None, 
    project_dir=None, results_path=None, pre=5, post=15, 
    min_frequency=0.005, min_duration=3, use_reindexed=True, 
    use_estimated_coords=False, skeleton=[], bodyparts=None, 
    use_bodyparts=None, num_samples=40, keypoint_colormap='autumn',
    plot_options={}, sampling_options={'mode':'density'},
    padding={'left':0.1, 'right':0.1, 'top':0.2, 'bottom':0.2},
    save_individually=True, save_gifs=True, save_mp4s=False, fps=30, 
    projection_planes=['xy','xz'], **kwargs):
    """
    Generate trajectory plots for a modeled dataset.

    Each trajectory plot shows a sequence of poses along the average
    trajectory through latent space associated with a given syllable.
    A separate figure (and gif, optionally) is saved for each syllable, 
    along with a single figure showing all syllables in a grid. The 
    plots are saved to ``{output_dir}`` if it is provided, otherwise 
    they are saved to ``{project_dir}/{name}/trajectory_plots``.

    Parameters
    ----------
    coordinates : dict, default=None
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). Required if
        ``use_estimated_coords=False``.

    results: dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.apply_model`). Must have
        the format::

            {
                session_name1: {
                    'syllables':              array of shape (n_frames,),
                    'estimated_coordinates' : array of shape (n_frames, n_bodyparts, dim)
                    'syllables_reindexed':    array of shape (n_frames,),
                    'centroid':               array of shape (n_frames, dim),
                    'heading' :               array of shape (n_frames,), 
                },
                ...  
            }
            
        - ``syllables`` is required if ``use_reindexed=False``
        - ``syllables_reindexed`` is required if ``use_reindexed=True``
        - ``centroid`` is always required
        - ``heading`` is always required
        - ``estimated_coordinates`` is required if ``use_estimated_coords=True``

        If ``results=None``, results will be loaded using either 
        ``results_path`` or  ``project_dir`` and ``name``.

    output_dir: str, default=None
        Directory where trajectory plots should be saved. If None, 
        plots will be saved to ``{project_dir}/{name}/trajectory_plots``.

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save trajectory
        plots if ``output_dir`` is None.

    project_dir: str, default=None
       Project directory. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save trajectory
        plots if ``output_dir`` is None.

    results_path: str, default=None
        Path to a results file. If None, results will be loaded from
        ``{project_dir}/{name}/results.h5``.

    pre: int, default=5, post: int, default=15
        Defines the temporal window around syllable onset for 
        computing the average trajectory. Note that the window is 
        independent of the actual duration of the syllable.

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to plotted.

    min_duration: float, default=3
        Minimum duration of a syllable instance to be included in the
        trajectory average.

    use_reindexed: bool, default=True
        Whether to use label syllables by their frequency rank (True) or
        or their original label (False). When reindexing, "0"  represents
        the most frequent syllable).

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. 

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts to include in trajectory plot.
        If None, all bodyparts will be included.

    skeleton : list, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.

    num_samples: int, default=40
        Number of samples to used to compute the average trajectory.
        Also used to set ``n_neighbors`` when sampling syllable instances
        in ``density`` mode. 

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    plot_options: dict, default={}
        Dictionary of options for trajectory plots (see
        :py:func:`keypoint_moseq.util.plot_trajectories`).

    sampling_options: dict, default={'mode':'density'}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).
        
    padding: dict, default={'left':0.1, 'right':0.1, 'top':0.2, 'bottom':0.2}
        Padding around trajectory plots. Controls the the distance
        between trajectories (when multiple are shown in one figure)
        as well as the title offset. 

    save_individually: bool, default=True
        If True, a separate figure is saved for each syllable (in
        addition to the grid figure).
        
    save_gifs: bool, default=True
        Whether to save an animated gif of the trajectory plots.
        
    save_mp4s: bool, default=False
        Whether to save videos of the trajectory plots as .mp4 files
        
    fps: int, default=30
        Framerate of the videos from which keypoints were derived.
        Used to set the framerate of gifs when ``save_gif=True``.
        
    projection_planes: list (subset of ['xy', 'yz', 'xz']), default=['xy','xz']
        For 3D data, defines the 2D plane(s) on which to project keypoint 
        coordinates. A separate plot will be saved for each plane with 
        the name of the plane (e.g. 'xy') as a suffix. This argument is 
        ignored for 2D data.
    """
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where trajectory plots '
            'should be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'trajectory_plots')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Saving trajectory plots to {output_dir}')
        
    if results is None: results = load_results(
        name=name, project_dir=project_dir, path=results_path)

    if use_estimated_coords:
        coordinates = {k:v['estimated_coordinates'] for k,v in results.items()}
    elif bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    syllable_key = 'syllables' + ('_reindexed' if use_reindexed else '')
    syllables = {k:v[syllable_key] for k,v in results.items()}
    centroids = {k:v['centroid'] for k,v in results.items()}
    headings = {k:v['heading'] for k,v in results.items()}
    plot_options.update({'keypoint_colormap':keypoint_colormap})
            
    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=num_samples)
    
    if len(syllable_instances) == 0:
        warnings.warn(fill(
            'No syllables with sufficient instances to make a trajectory plot. '
            'This usually occurs when all frames have the same syllable label '
            '(use `plot_syllable_frequencies` to check if this is the case)'))
        return
    
    sampling_options['n_neighbors'] = num_samples
    sampled_instances = sample_instances(
        syllable_instances, num_samples, coordinates=coordinates, 
        centroids=centroids, headings=headings, **sampling_options)

    trajectories = {syllable: get_trajectories(
        instances, coordinates, pre=pre, post=post, 
        centroids=centroids, headings=headings
        ) for syllable,instances in sampled_instances.items()}

    edges = []
    if len(skeleton)>0: 
        if isinstance(skeleton[0][0],str):
            assert use_bodyparts is not None, fill(
                'If skeleton edges are specified using bodypart names, '
                '``use_bodyparts`` must be specified')
            edges = get_edges(use_bodyparts, skeleton)
        else: edges = skeleton

    syllables = sorted(trajectories.keys())
    titles = [f'Syllable {syllable}' for syllable in syllables]
    Xs = np.nanmean(np.array([trajectories[syllable] for syllable in syllables]),axis=1)  
    
    if Xs.shape[-1]==3:
        projection_planes = [''.join(sorted(plane.lower())) for plane in projection_planes]
        assert set(projection_planes) <= set(['xy','yz','xz']), fill(
            "`projection_planes` must be a subset of `['xy','yz','xz']`")
        all_Xs = [Xs[...,np.array({'xy':[0,1], 'yz':[1,2], 'xz':[0,2]}[plane])] for plane in projection_planes]
        suffixes = ['.'+plane for plane in projection_planes]
       
    else: 
        all_Xs = [Xs]
        suffixes = ['']

    for Xs,suffix in zip(all_Xs,suffixes):
        lims = get_limits(Xs, pctl=0, **padding)

        # individual plots
        if save_individually:
            desc = 'Generating trajectory plots'
            for title,X in tqdm.tqdm(zip(titles,Xs), desc=desc, total=len(titles)):

                fig,ax,rasters = plot_trajectories(
                    [title], X[None], lims, edges=edges, 
                    return_rasters=(save_gifs or save_mp4s),
                    **plot_options)

                plt.savefig(os.path.join(output_dir, f'{title}{suffix}.pdf'))
                plt.close(fig=fig)

                if save_gifs:
                    use_fps = len(rasters)/(pre+post)*fps
                    path = os.path.join(output_dir, f'{title}{suffix}.gif')
                    imageio.mimsave(path, rasters, fps=use_fps)
                if save_mp4s:
                    use_fps = len(rasters)/(pre+post)*fps
                    path = os.path.join(output_dir, f'{title}{suffix}.mp4')
                    write_video_clip(rasters, path, fps=use_fps)
                    

        # grid plot
        fig,ax,rasters = plot_trajectories(
            titles, Xs, lims, edges=edges, 
            return_rasters=(save_gifs or save_mp4s),
            **plot_options)

        plt.savefig(os.path.join(output_dir, f'all_trajectories{suffix}.pdf'))
        plt.show()

        if save_gifs:
            use_fps = len(rasters)/(pre+post)*fps
            path = os.path.join(output_dir, f'all_trajectories{suffix}.gif')
            imageio.mimsave(path, rasters, fps=use_fps)
        if save_mp4s:
            use_fps = len(rasters)/(pre+post)*fps
            path = os.path.join(output_dir, f'all_trajectories{suffix}.mp4')
            write_video_clip(rasters, path, fps=use_fps)



def overlay_keypoints_on_image(
    image, coordinates, edges=[], keypoint_colormap='autumn',
    node_size=3, line_width=2, copy=False, opacity=1.0):
    """
    Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.
    
    coordinates: ndarray of shape (num_keypoints, 2)
        Array of keypoint coordinates.

    edges: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of indexes.

    keypoint_colormap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    node_size: int, default=10
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.
    
    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity<1.0: 
        canvas = image.copy()
    else: canvas = image

    # get colors from matplotlib and convert to 0-255 range for openc
    colors = plt.get_cmap(keypoint_colormap)(np.linspace(0,1,coordinates.shape[0]))
    colors = [tuple([int(c) for c in cs[:3]*255]) for cs in colors]

    # overlay skeleton
    for i, j in edges:
        if np.isnan(coordinates[i,0]) or np.isnan(coordinates[j,0]): continue
        pos1 = tuple(coordinates[i].astype(int))
        pos2 = tuple(coordinates[j].astype(int))
        canvas = cv2.line(canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA)

    # overlay keypoints
    for i, (x,y) in enumerate(coordinates):
        if np.isnan(x) or np.isnan(y): continue
        pos = (int(x), int(y))
        canvas = cv2.circle(canvas, pos, node_size, colors[i], -1, lineType=cv2.LINE_AA)

    if opacity<1.0:
        image = cv2.addWeighted(image, 1-opacity, canvas, opacity, 0)
    return image


def overlay_trajectory_on_video(
        frames, trajectory, smoothing_kernel=1, highlight=None, 
        min_opacity=0.2, max_opacity=1, num_ghosts=10, interval=2, 
        plot_options={}):
    
    """
    Overlay a trajectory of keypoints on a video.
    """
    if smoothing_kernel > 0:
        trajectory = gaussian_filter1d(trajectory, smoothing_kernel, axis=0)
        
    opacities = np.repeat(np.linspace(max_opacity, min_opacity, num_ghosts+1), interval)  
    for i in np.arange(0,trajectory.shape[0],interval):
        for j,opacity in enumerate(opacities):
            if i+j < frames.shape[0]:
                plot_options['opacity'] = opacity
                if highlight is not None:
                    start,end,highlight_factor = highlight
                    if i+j < start or i+j > end:
                        plot_options['opacity'] *= highlight_factor
                frames[i+j] = overlay_keypoints_on_image(
                    frames[i+j], trajectory[i], **plot_options)
    return frames

def overlay_keypoints_on_video(
    video_path, coordinates, skeleton=[], bodyparts=None, use_bodyparts=None, 
    output_path=None, show_frame_numbers=True, text_color=(255,255,255), 
    crop_size=None, frames=None, quality=7, centroid_smoothing_filter=10, 
    plot_options={}):
    """
    Overlay keypoints on a video.

    Parameters
    ----------
    video_path: str
        Path to a video file.

    coordinates: ndarray of shape (num_frames, num_keypoints, 2)
        Array of keypoint coordinates.

    skeleton: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. Required if
        ``skeleton`` is defined using bodypart names.

    use_bodyparts: list of str, default=None
        Subset of bodyparts to plot. If None, all bodyparts are plotted.

    output_path: str, default=None
        Path to save the video. If None, the video is saved to
        ``video_path`` with the suffix ``_keypoints``.

    show_frame_numbers: bool, default=True
        Whether to overlay the frame number in the video.

    text_color: tuple of int, default=(255,255,255)
        Color for the frame number overlay.

    crop_size: int, default=None
        Size of the crop around the keypoints to overlay on the video.
        If None, the entire video is used. 

    frames: iterable of int, default=None
        Frames to overlay keypoints on. If None, all frames are used.

    quality: int, default=7
        Quality of the output video.

    centroid_smoothing_filter: int, default=10
        Amount of smoothing to determine cropping centroid.

    plot_options: dict, default={}
        Additional keyword arguments to pass to
        :py:func:`keypoint_moseq.viz.overlay_keypoints`.
    """
    if output_path is None: 
        output_path = os.path.splitext(video_path)[0] + '_keypoints.mp4'

    if bodyparts is not None:
        if use_bodyparts is not None:
            coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)
        else: use_bodyparts = bodyparts
        edges = get_edges(use_bodyparts, skeleton)
    else: edges = skeleton

    if crop_size is not None:
        outliers = np.any(np.isnan(coordinates), axis=2)
        interpolated_coordinates = interpolate_keypoints(coordinates, outliers)
        crop_centroid = np.nanmedian(interpolated_coordinates, axis=1)
        crop_centroid = gaussian_filter1d(crop_centroid, centroid_smoothing_filter, axis=0)

    with imageio.get_reader(video_path) as reader:
        fps = reader.get_meta_data()['fps']
        if frames is None: frames = np.arange(reader.count_frames())

        with imageio.get_writer(
            output_path, pixelformat='yuv420p', 
            fps=fps, quality=quality) as writer:

            for frame in tqdm.tqdm(frames):
                image = reader.get_data(frame)

                image = overlay_keypoints_on_image(
                    image, coordinates[frame], edges=edges, **plot_options)

                if crop_size is not None:
                    image = crop_image(image, crop_centroid[frame], crop_size)

                if show_frame_numbers:
                    image = cv2.putText(
                        image, f'Frame {frame}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, text_color, 1, cv2.LINE_AA)

                writer.append_data(image)





def crowd_movie(
    instances, coordinates, lims, pre=30, post=60,
    edges=[], plot_options={}):
    """
    Generate a crowd movie.

    Crowd movies show many instances of a syllable by animating
    their keypoint trajectories in a common coordinate system.
    The trajectories are synchronized to syllable onset. The opacity 
    of each instance increases at syllable onset and decreases at
    syllable offset.

    Parameters
    ----------
    instances: list of tuples ``(key, start, end)``
        List of syllable instances to include in the grid movie,
        where each instance is specified as a tuple with the session
        name, start frame and end frame. 
    
    coordinates: dict of ndarray of shape (num_frames, num_keypoints, dim)
        Dictionary of keypoint coordinates, where each key is a session name.

    lims: array of shape (2, dim)
        Axis limits for plotting keypoints in the crowd movies. 

    pre: int, default=30
        Number of frames before syllable onset to include in the movie

    post: int, default=60
        Number of frames after syllable onset to include in the movie

    plot_options: dict, default={}
        Dictionary of options for rendering keypoints in the crowd
        movies (see :py:func:`keypoint_moseq.util.overlay_keypoints_on_image`).

    Returns
    -------
    frames: array of shape ``(post+pre, height, width, 3)``
        Array of frames in the grid movie. ``width`` and 
        ``height`` are determined by ``lims``.
    """
    dim = coordinates[instances[0][0]].shape[2]
    if dim == 3: warnings.warn(fill(
        'Crowd movies are only supported for 2D keypoints. '
        'Only the X and Y coordinates will be used.'))

    h, w = (lims[1]-lims[0]).astype(int)
    frames = np.zeros((post+pre, w, h, 3), dtype=np.uint8)

    for key, start, end in instances:
        xy = coordinates[key][start-pre:start+post,:,:2]
        xy = (np.clip(xy, *lims[:,:2]) - lims[0,:2])
        frames = overlay_trajectory_on_video(
            frames, xy, plot_options=plot_options)

        dot_radius=5
        dot_color=(255,255,255)
        centroids = gaussian_filter1d(xy.mean(1),1,axis=0)
        for i in range(pre, min(end-start+pre,pre+post)):
            pos = (int(centroids[i,0]),int(centroids[i,1]))
            frames[i] = cv2.circle(frames[i], pos, dot_radius, dot_color, -1, cv2.LINE_AA)

    return frames


def generate_crowd_movies(
    coordinates, results=None, output_dir=None, name=None, 
    project_dir=None, results_path=None, pre=30, post=60,
    min_frequency=0.005, min_duration=3, num_instances=15,
    use_reindexed=True, use_estimated_coords=False, skeleton=[], 
    bodyparts=None, use_bodyparts=None, keypoint_colormap='autumn', 
    fps=30, limits=None, plot_options={}, sampling_options={}, 
    quality=7, **kwargs):
    """
    Generate crowd movies for a modeled dataset.

    Crowd movies show many instances of a syllable and are useful in
    figuring out what behavior the syllable captures 
    (see :py:func:`keypoint_moseq.viz.crowd_movie`). This method
    generates a crowd movie for each syllable that is used sufficiently
    often (i.e. has at least ``num_instances`` instances with duration
    of at least ``min_duration`` and an overall frequency of at least
    ``min_frequency``). The crowd movies are saved to ``output_dir`` if 
    specified, or else to ``{project_dir}/{name}/crowd_movies``.

    Parameters
    ----------
    coordinates: dict, default=None
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). Required if
        ``use_estimated_coords=False``.

    results: dict, default=None
        Dictionary containing modeling results for a dataset (see
        :py:func:`keypoint_moseq.fitting.apply_model`). Must have
        the format::

            {
                session_name1: {
                    'syllables':              array of shape (n_frames,),
                    'estimated_coordinates' : array of shape (n_frames, n_bodyparts, dim)
                    'syllables_reindexed':    array of shape (n_frames,),
                    'centroid':               array of shape (n_frames, dim),
                    'heading' :               array of shape (n_frames,), 
                },
                ...  
            }
            
        - ``syllables`` is required if ``use_reindexed=False``
        - ``syllables_reindexed`` is required if ``use_reindexed=True``
        - ``centroid`` is required if the sampling mode is 'density'
        - ``heading`` is required if the sampling mode is 'density'
        - ``estimated_coordinates`` is required if ``use_estimated_coords=True``
        

        If ``results=None``, results will be loaded using either 
        ``results_path`` or  ``project_dir`` and ``name``.

    output_dir: str, default=None
        Directory where crowd movies should be saved. If None, 
        movies will be saved to ``{project_dir}/{name}/crowd_movies``.

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save crowd
        movies if ``output_dir`` is None.

    project_dir: str, default=None
        Project directory. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save crowd
        movies if ``output_dir`` is None.

    results_path: str, default=None
        Path to a results file. If None, results will be loaded from
        ``{project_dir}/{name}/results.h5``.

    num_instances: int, default=15
        Number of syllable instances per crowd movie.

    min_frequency: float, default=0.005
        Minimum frequency of a syllable to be included in the crowd movies.

    min_duration: int, default=3
        Minimum duration of a syllable instance to be included in the 
        crowd movie for that syllable. 

    use_reindexed: bool, default=True
        Whether to use label syllables by their frequency rank (True) or
        or their original label (False). When reindexing, "0"  represents
        the most frequent syllable).

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. 

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts to include in the crowd
        movies. If None, all bodyparts will be included.
        
    skeleton: list, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names or a pair of indexes.
        
    keypoint_colormap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.
        
    fps: int, default=30
        Frames per second of the crowd movies.

    limits: array, default=None
        Axis limits for plotting keypoints in the crowd movies. If None,
        limits will be inferred automatically from ``coordinates``.
        
    plot_options: dict, default={}
        Dictionary of options for rendering keypoints in the crowd
        movies (see :py:func:`keypoint_moseq.util.overlay_keypoints_on_image`).
        
    sampling_options: dict, default={}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).

    quality: int, default=7
        Quality of the crowd movies. Higher values result in higher
        quality movies but larger file sizes.
    """    
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where crowd movies should '
            'be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'crowd_movies')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Writing crowd movies to {output_dir}')

    if results is None: results = load_results(
        name=name, project_dir=project_dir, path=results_path)
        
    if use_estimated_coords:
        coordinates = {k:v['estimated_coordinates'] for k,v in results.items()}
    elif bodyparts is not None and use_bodyparts is not None:
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    syllable_key = 'syllables' + ('_reindexed' if use_reindexed else '')
    syllables = {k:v[syllable_key] for k,v in results.items()}
    plot_options.update({'keypoint_colormap':keypoint_colormap})

    if limits is None: 
        limits = get_limits(coordinates, blocksize=16)

    centroids,headings = None,None
    k = list(results.keys())[0]
    if 'centroid' in results[k]:
        centroids = {k:v['centroid'] for k,v in results.items()}
    if 'heading' in results[k]:
        headings = {k:v['heading'] for k,v in results.items()}

    edges = []
    if len(skeleton)>0: 
        if isinstance(skeleton[0][0],str):
            assert use_bodyparts is not None, fill(
                'If skeleton edges are specified using bodypart names, '
                '``use_bodyparts`` must be specified')
            edges = get_edges(use_bodyparts, skeleton)
        else: edges = skeleton

    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=num_instances)
    
    if len(syllable_instances) == 0:
        warnings.warn(fill(
            'No syllables with sufficient instances to make a crowd movie. '
            'This usually occurs when all frames have the same syllable label '
            '(use `plot_syllable_frequencies` to check if this is the case)'))
        return

    sampled_instances = sample_instances(
        syllable_instances, num_instances, coordinates=coordinates, 
        centroids=centroids, headings=headings, **sampling_options)

    for syllable,instances in tqdm.tqdm(
        sampled_instances.items(), desc='Generating crowd movies'):
        
        frames = crowd_movie(
            instances, coordinates, pre=pre, post=post,
            edges=edges, lims=limits, plot_options=plot_options)

        path = os.path.join(output_dir, f'syllable{syllable}.mp4')
        write_video_clip(frames, path, fps=fps, quality=quality)
            