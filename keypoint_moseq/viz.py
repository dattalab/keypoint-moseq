import os
import cv2
import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
plt.rcParams['figure.dpi'] = 100

from vidio.read import OpenCVReader
from textwrap import fill

from keypoint_moseq.util import (
    get_edges, get_durations, get_frequencies, reindex_by_bodyparts,
    find_matching_videos, get_syllable_instances, sample_instances,
    filter_centroids_headings, get_trajectories, interpolate_keypoints
)
from keypoint_moseq.io import load_results
from jax_moseq.models.keypoint_slds import center_embedding




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
             savefig=True, project_dir=None, scale=10, plot_n_pcs=10, 
             axis_size=(2,1.5), ncols=5, node_size=20, **kwargs):
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
        saved to ``{project_dir}/pcs-{xy/yz}.pdf`` (``yz`` is only 
        included for 3D data).

    project_dir : str, default=None
        Path to the project directory. Required if ``savefig`` is True.

    scale : float, default=10
        Scale factor for the perturbation of the mean pose.

    plot_n_pcs : int, default=10
        Number of PCs to plot. 

    axis_size : tuple of float, default=(2,1.5)
        Size of each subplot in inches.

    ncols : int, default=5
        Number of columns in the figure.

    node_size : int, default=20
        Size of the keypoints in the figure.
    """
    k = len(use_bodyparts)
    d = len(pca.mean_)//(k-1)  
    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    cmap = plt.cm.get_cmap(keypoint_colormap)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])
    
    if d==2: dims_list,names = [[0,1]],['xy']
    if d==3: dims_list,names = [[0,1],[1,2]],['xy','yz']
    
    for dims,name in zip(dims_list,names):
        nrows = int(np.ceil(plot_n_pcs/ncols))
        fig,axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i,ax in enumerate(axs.flat):
            ymean = Gamma @ pca.mean_.reshape(k-1,d)[:,dims]
            y = Gamma @ (pca.mean_ + scale*pca.components_[i]).reshape(k-1,d)[:,dims]
            for e in edges: ax.plot(*ymean[e].T, color=cmap(e[0]/(k-1)), zorder=0, alpha=0.25)
            ax.scatter(*ymean.T, c=np.arange(k), cmap=cmap, s=node_size, zorder=1, alpha=0.25, linewidth=0)
            for e in edges: ax.plot(*y[e].T, color=cmap(e[0]/(k-1)), zorder=2)
            ax.scatter(*y.T, c=np.arange(k), cmap=cmap, s=node_size, zorder=3)
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
        

def plot_progress(model, data, history, iteration, path=None,
                  project_dir=None, name=None, savefig=True,
                  fig_size=None, seq_length=600, min_frequency=.001, 
                  **kwargs):
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

    project_dir : str, default=None
    name : str, default=None
    path : str, default=None
    """
    z = np.array(model['states']['z'])
    mask = np.array(data['mask'])
    durations = get_durations(z,mask)
    frequencies = get_frequencies(z,mask)
    
    history_iters = np.sort(history.keys())
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
    axs[0].bar(range(len(frequencies)),frequencies,width=1)
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('syllable rank')
    axs[0].set_title('Frequency distribution')
    axs[0].set_yticks([])
    
    lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim/30)),1)
    lim = lim-(lim%binsize)
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

        yticks = axs[2].get_yticks()
        yticklabels = history_iters[yticks.astype(int)]
        axs[2].set_yticklabels(yticklabels)
        
        nz = np.stack(np.array(mask[:,seq_length:]).nonzero(),axis=1)
        batch_ix,start = nz[np.random.randint(nz.shape[0])]
        seq_hist = np.stack([z[batch_ix,start:start+seq_length] for z in past_stateseqs])
        axs[3].imshow(seq_hist, cmap=plt.cm.jet, aspect='auto', interpolation='nearest')
        axs[3].set_xlabel('Time (frames)')
        axs[3].set_ylabel('Iterations')
        axs[3].set_title('State sequence history')
        
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


def _crowd_movie_tile(key, start, end, videos, centroids, headings, 
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
            if 0 <= ii-pre <= end-start:
                pos = tuple([int(x) for x in M@np.append(c,1)])
                cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            tile.append(frame)  
        return np.stack(tile)
    
    
def crowd_movie(instances, rows, cols, videos, centroids, headings,
                dot_color=(255,255,255), dot_radius=4, window_size=112, 
                pre=30, post=60):
    
    """Generate a crowd movie and return it as an array of frames.

    Crowd movies show many instances of a syllable. Each instance
    shows a snippet of a video, centered on the animal and synchronized
    to the onset of the syllable. A dot appears at syllable onset and 
    disappears at syllable offset.

    Parameters
    ----------
    instances: list of tuples ``(key, start, end)``
        List of syllable instances to include in the crowd movie,
        where each instance is specified as a tuple with the video 
        name, start frame and end frame. The list must have length
        ``rows*cols``. The video names must also be keys in ``videos``.
        
    rows: int, cols : int
        Number of rows and columns in the crowd movie grid
    
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
        Size of the window around the animal

    pre: int, default=30
        Number of frames before syllable onset to include in the movie

    post: int, default=60
        Number of frames after syllable onset to include in the movie

    Returns
    -------
    frames: array of shape ``(rows, cols, post+pre, window_size, window_size, 3)``
        Array of frames in the crowd movie
    """
    tiles = np.stack([
        _crowd_movie_tile(
            key, start, end, videos, centroids, headings, 
            dot_color=dot_color, window_size=window_size,
            pre=pre, post=post, dot_radius=dot_radius
        ) for key, start, end in instances
    ]).reshape(rows, cols, post+pre, window_size, window_size, 3)
    return np.concatenate(np.concatenate(tiles,axis=2),axis=2)


    
def write_video_clip(frames, path, fps=30, quality=7):
    """Write a video clip to a file.
    """
    with imageio.get_writer(
        path, pixelformat='yuv420p', 
        fps=fps, quality=quality) as writer:
        for frame in frames: 
            writer.append_data(frame)


def generate_crowd_movies(
    results=None, output_dir=None, name=None, project_dir=None,
    results_path=None, video_dir=None, video_paths=None, 
    rows=4, cols=6, filter_size=9, pre=30, post=60, 
    min_frequency=0.005, min_duration=3, dot_radius=4, 
    dot_color=(255,255,255), window_size=112, use_reindexed=True, 
    sampling_options={}, coordinates=None, bodyparts=None, 
    use_bodyparts=None, quality=7, video_extension=None, **kwargs):
    
    """
    Generate crowd movies for a modeled dataset.

    Crowd movies show many instances of a syllable and are useful in
    figuring out what behavior the syllable captures 
    (see :py:func:`keypoint_moseq.viz.crowd_movie`). This method
    generates a crowd movie for each syllable that is used sufficiently
    often (i.e. has at least ``rows_cols`` instances with duration
    of at least ``min_duration`` and an overall frequency of at least
    ``min_frequency``). The crowd movies are saved to ``output_dir`` if 
    specified, or else to ``{project_dir}/{name}/crowd_movies``.

    Parameters
    ----------
    results: dict, default=None
        Dictionary containing modeling results for a dataset. Must 
        contain syllable sequences, centroids and headings (see
        :py:func:`keypoint_moseq.fitting.apply_model`). If None,
        results will be loaded using either ``results_path`` or 
        ``project_dir`` and ``name``.

    output_dir: str, default=None
        Directory where crowd movies should be saved. If None, crowd
        movies will be saved to ``{project_dir}/{name}/crowd_movies``.

    name: str, default=None
        Name of the model. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save crowd movies 
        if ``output_dir`` is None.
        
    project_dir: str, default=None
        Project directory. Required to load results if ``results`` is 
        None and ``results_path`` is None. Required to save crowd movies 
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
        Minimum frequency of a syllable to be included in the crowd movies.

    min_duration: int, default=3
        Minimum duration of a syllable instance to be included in the 
        crowd movie for that syllable. 

    use_reindexed: bool, default=True
        Determines the naming of syllables (``results["syllables"]`` if 
        False, or ``results["syllables_reindexed"]`` if True). The
        reindexed naming corresponds to the rank order of syllable
        frequency (e.g. "0" for the most frequent syllable).

    sampling_options: dict, default={}
        Dictionary of options for sampling syllable instances (see
        :py:func:`keypoint_moseq.util.sample_instances`).
    
    coordinates: dict, default=None
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). Required
        for density-based sampling (i.e. when 
        ``sampling_options['mode']=='density'``; see 
        :py:func:`keypoint_moseq.util.sample_instances`).

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. Required when 
        ``coordinates`` is provided and bodyparts were reindexed 
        for modeling. 

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts used for modeling. Required when 
        ``coordinates`` is provided and bodyparts were reindexed 
        for modeling. 

    quality: int, default=7
        Quality of the crowd movies. Higher values result in higher
        quality movies but larger file sizes.

    rows, cols, pre, post, dot_radius, dot_color, window_size
        See :py:func:`keypoint_moseq.viz.crowd_movie`

    video_extension: str, default=None
        Preferred video extension (passed to :py:func:`keypoint_moseq.util.find_matching_videos`)
    """
    assert (video_dir is not None) or (video_paths is not None), fill(
        'You must provide either ``video_dir`` or ``video_paths``')      
    
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where crowd movies should '
            'be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'crowd_movies')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Writing crowd movies to {output_dir}')
    
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

    sampled_instances = sample_instances(
        syllable_instances, rows*cols, coordinates=coordinates, 
        centroids=centroids, headings=headings, **sampling_options)

    centroids,headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size)
    
    for syllable,instances in tqdm.tqdm(
        sampled_instances.items(), desc='Generating crowd movies'):
        
        frames = crowd_movie(
            instances, rows, cols, videos, centroids, headings, 
            window_size=window_size, dot_color=dot_color, 
            dot_radius=dot_radius, pre=pre, post=post)

        path = os.path.join(output_dir, f'syllable{syllable}.mp4')
        write_video_clip(frames, path, fps=fps, quality=quality)

        
        

def _pad_limits(limits, left=0.1, right=0.1, top=0.1, bottom=0.1):
    
    xmin,ymin = limits[0]
    xmax,ymax = limits[1]
    width = xmax-xmin
    height = ymax-ymin
    
    xmin -= width*left
    xmax += width*right
    ymin -= height*bottom
    ymax += height*top
    
    return np.array([
        [xmin,ymin],
        [xmax,ymax]])


        
def plot_trajectories(titles, Xs, edges, lims, n_cols=4, invert=False, 
                      keypoint_colormap='autumn', node_size=50, linewidth=3, 
                      plot_width=4, overlap=(0.2,0)):
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

    edges: list of tuples
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

    keypoint_colormap : str
        Name of a matplotlib colormap to use for coloring the keypoints.

    node_size: int, default=50
        Size of each keypoint.

    linewidth: int, default=3
        Width of the lines connecting keypoints.

    plot_width: int, default=4
        Width of each trajectory plot in inches. The height  is 
        determined by the aspect ratio of ``lims``. The final figure 
        width is ``fig_width * min(n_cols, len(X))``.

    overlap: tuple of float, default=(0.2,0)
        Amount of overlap between each trajectory plot as a tuple 
        with the format ``(x_overlap, y_overlap)``. The values should
        be between 0 and 1.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Figure handle

    ax: matplotlib.axes.Axes
        Axis containing the trajectory plots.
    """
    num_timesteps = Xs[0].shape[0]
    num_keypoints = Xs[0].shape[1]

    interval = int(np.floor(num_timesteps/10))
    plot_frames = np.arange(0,num_timesteps,interval)
    colors = plt.cm.get_cmap(keypoint_colormap)(np.linspace(0,1,num_keypoints))
    fill_color = 'k' if invert else 'w'

    n_cols = min(n_cols, len(Xs))
    n_rows = np.ceil(len(Xs)/n_cols)
    offsets = np.stack(np.meshgrid(
        np.arange(n_cols)*np.diff(lims[:,0])*(1-overlap[0]),
        np.arange(n_rows)*np.diff(lims[:,1])*(overlap[1]-1)
    ),axis=-1).reshape(-1,2)[:len(Xs)]
    
    Xs = np.array(Xs)+offsets[:,None,None]
    xmin,ymin = lims[0] + offsets.min(0)
    xmax,ymax = lims[1] + offsets.max(0)

    
    fig,ax = plt.subplots()

    ax.fill_between(
        [xmin,xmax], y1=[ymax,ymax], y2=[ymin,ymin], 
        facecolor=fill_color, zorder=0, clip_on=False)
        
    for i in plot_frames:
        
        for X,offset in zip(Xs,offsets):
            for ii,jj in edges: 
                ax.plot(*X[i,(ii,jj)].T, c='k', zorder=i*4, 
                        linewidth=linewidth, clip_on=False)
        
            for ii,jj in edges: 
                ax.plot(*X[i,(ii,jj)].T, c=colors[ii], zorder=i*4+1, 
                        linewidth=linewidth*.9, clip_on=False)

            ax.scatter(*X[i].T, c=colors, zorder=i*4+2, edgecolor='k', 
                       linewidth=0.4, s=node_size, clip_on=False)
        
        if i < plot_frames.max(): 
            ax.fill_between(
                [xmin,xmax], y1=[ymax,ymax], y2=[ymin,ymin], 
                facecolor=fill_color, alpha=0.2, zorder=i*4+3, clip_on=False)

            
    title_xy = (lims * np.array([[0.5,0.1],[0.5,0.9]])).sum(0)
    title_color = 'w' if invert else 'k'

    for xy,text in zip(offsets+title_xy,titles):
        ax.text(*xy, text, c=title_color, ha='center', 
                va='top', zorder=plot_frames.max()*4+4)
        
    plot_height = plot_width*(ymax-ymin)/(xmax-xmin)*1.1
    fig_width = plot_width*n_cols - (n_cols-1)*plot_width*overlap[0]
    fig_height = plot_height*n_rows - (n_rows-1)*plot_height*overlap[1]
    fig.set_size_inches((fig_width, fig_height))
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig,ax
    
    


def generate_trajectory_plots(
    coordinates, results=None, output_dir=None, name=None, 
    project_dir=None, results_path=None, pre=5, post=15, 
    min_frequency=0.005, min_duration=3, use_reindexed=True, 
    skeleton=None, bodyparts=None, use_bodyparts=None,  
    num_samples=40, keypoint_colormap='autumn',
    plot_options={}, sampling_options={'mode':'density'}, **kwargs):
    """
    Generate trajectory plots for a modeled dataset.

    Each trajectory plot shows a sequence of poses along the average
    trajectory through latent space associated with a given syllable.
    A separate figure is saved for each syllable, along with a single
    figure showing all syllables in a grid. The plots are saved to
    ``{output_dir}`` if it is provided, otherwise they are saved to
    ``{project_dir}/{name}/trajectory_plots``.

    Parameters
    ----------
    coordinates : dict
        Dictionary mapping session names to keypoint coordinates as 
        ndarrays of shape (n_frames, n_bodyparts, 2). 

    results: dict, default=None
        Dictionary containing modeling results for a dataset. Must 
        contain syllable sequences, centroids and headings (see
        :py:func:`keypoint_moseq.fitting.apply_model`). If None,
        results will be loaded using either ``results_path`` or 
        ``project_dir`` and ``name``.

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
        Determines the naming of syllables (``results["syllables"]`` if 
        False, or ``results["syllables_reindexed"]`` if True). The
        reindexed naming corresponds to the rank order of syllable
        frequency (e.g. "0" for the most frequent syllable).

    bodyparts: list of str, default=None
        List of bodypart names in ``coordinates``. 

    use_bodyparts: list of str, default=None
        Ordered list of bodyparts that were used for modeling.

    skeleton : list
        List of edges that define the skeleton, where each edge is a
        pair of bodypart names.

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

    """
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where trajectory plots '
            'should be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'trajectory_plots')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Saving trajectory plots to {output_dir}')
    
    if not (bodyparts is None or use_bodyparts is None):
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)
    
    if results is None: results = load_results(
        name=name, project_dir=project_dir, path=results_path)

    syllable_key = 'syllables' + ('_reindexed' if use_reindexed else '')
    syllables = {k:v[syllable_key] for k,v in results.items()}
    centroids = {k:v['centroid'] for k,v in results.items()}
    headings = {k:v['heading'] for k,v in results.items()}
    plot_options.update({'keypoint_colormap':keypoint_colormap})
            
    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=num_samples)
    
    sampling_options['n_neighbors'] = num_samples
    sampled_instances = sample_instances(
        syllable_instances, num_samples, coordinates=coordinates, 
        centroids=centroids, headings=headings, **sampling_options)

    trajectories = get_trajectories(
        sampled_instances, coordinates, pre=pre, post=post, 
        centroids=centroids, headings=headings)

    edges = []
    if skeleton is not None: 
        assert use_bodyparts is not None, fill(
            'To plot skeleton edges, ``use_bodyparts`` must be specified')
        edges = get_edges(use_bodyparts, skeleton)

    syllables = sorted(trajectories.keys())
    titles = [f'Syllable {syllable}' for syllable in syllables]
    Xs = np.array([trajectories[syllable] for syllable in syllables]).mean(1)
    
    lims = np.stack([Xs.min((0,1,2)),Xs.max((0,1,2))])
    lims = _pad_limits(lims, left=0.1, right=0.1, top=0.2, bottom=0.2)

    if Xs.shape[-1]==2:
        
        # individual plots
        desc = 'Generating trajectory plots'
        for title,X in tqdm.tqdm(zip(titles,Xs), desc=desc, total=len(titles)):
            fig,ax = plot_trajectories([title], X[None], edges, lims, **plot_options)
            path = os.path.join(output_dir, f'{title}.pdf')
            plt.savefig(path)
            plt.close(fig=fig)

        # grid plot
        fig,ax = plot_trajectories(titles, Xs, edges, lims, **plot_options)
        path = os.path.join(output_dir, 'all_trajectories.pdf')
        plt.savefig(path)
        plt.show()
            
    else: raise NotImplementedError()


def overlay_keypoints(
    image, coordinates, skeleton_idx=[], keypoint_colormap='autumn',
    node_size=10, line_width=2, copy=False):
    """
    Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.
    
    coordinates: ndarray of shape (num_keypoints, 2)
        Array of keypoint coordinates.

    skeleton_idx: list of tuples, default=[]
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

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy: image = image.copy()

    # get colors from matplotlib and convert to 0-255 range for openc
    colors = plt.get_cmap(keypoint_colormap)(np.linspace(0,1,coordinates.shape[0]))
    colors = [tuple([int(c) for c in cs[:3]*255]) for cs in colors]

    # overlay skeleton
    for i, j in skeleton_idx:
        if np.isnan(coordinates[i,0]) or np.isnan(coordinates[j,0]): continue
        pos1 = tuple(coordinates[i].astype(int))
        pos2 = tuple(coordinates[j].astype(int))
        image = cv2.line(image, pos1, pos2, colors[i], line_width, cv2.LINE_AA)

    # overlay keypoints
    for i, (x,y) in enumerate(coordinates):
        if np.isnan(x): continue
        pos = (int(x), int(y))
        image = cv2.circle(image, pos, node_size, colors[i], -1, cv2.LINE_AA)

    return image

def crop_image(image, centroid, crop_size):
    """
    Crop an image around a centroid.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to crop.

    centroid: tuple of int
        (x,y) coordinates of the centroid.

    crop_size: int
        Size of the crop around the centroid.

    Returns
    -------
    image: ndarray of shape (crop_size, crop_size, 3)
        Cropped image.
    """
    x, y = centroid
    x = int(np.clip(x, crop_size, image.shape[1]-crop_size))
    y = int(np.clip(y, crop_size, image.shape[0]-crop_size))
    crop_size = int(crop_size)
    return image[y-crop_size:y+crop_size, x-crop_size:x+crop_size]




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
        skeleton_idx = get_edges(use_bodyparts, skeleton)
    else: skeleton_idx = skeleton

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

                image = overlay_keypoints(
                    image, coordinates[frame], skeleton_idx=skeleton_idx, **plot_options)

                if crop_size is not None:
                    image = crop_image(image, crop_centroid[frame], crop_size)

                if show_frame_numbers:
                    image = cv2.putText(
                        image, f'Frame {frame}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, text_color, 1, cv2.LINE_AA)

                writer.append_data(image)
