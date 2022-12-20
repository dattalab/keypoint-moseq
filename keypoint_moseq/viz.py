import os
import cv2
import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

from vidio.read import OpenCVReader
from textwrap import fill

from keypoint_moseq.util import (
    get_edges, get_durations, get_frequencies, reindex_by_bodyparts,
    find_matching_videos, get_syllable_instances, sample_instances,
    filter_centroids_headings, get_trajectories,
)
from keypoint_moseq.io import load_results
from jax_moseq.models.keypoint_slds import center_embedding




def plot_scree(pca, savefig=True, project_dir=None):
    fig = plt.figure()
    plt.plot(np.arange(len(pca.mean_))+1,np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('PCs')
    plt.ylabel('Explained variance')
    plt.yticks(np.arange(0.5,1.01,.1))
    plt.xticks(range(0,len(pca.mean_)+2,2))
    plt.gcf().set_size_inches((2.5,2))
    plt.grid()
    plt.tight_layout()
    
    if savefig:
        assert project_dir is not None, fill(
            'The ``savefig`` option requires a ``project_dir``')
        plt.savefig(os.path.join(project_dir,'pca_scree.pdf'))
    plt.show()
          
def plot_pcs(pca, *, use_bodyparts, skeleton, keypoint_colormap,
             savefig=True, project_dir=None, scale=10, plot_n_pcs=10, 
             axis_size=(2,1.5), ncols=5, node_size=20, **kwargs):
    
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
    
    z = np.array(model['states']['z'])
    mask = np.array(data['mask'])
    durations = get_durations(z,mask)
    frequencies = get_frequencies(z,mask)
    
    history_iters = sorted(history.keys())
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
    axs[0].set_title('Usage distribution')
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
        axs[2].set_title('median duration')
        
        nz = np.stack(np.array(mask[:,seq_length:]).nonzero(),axis=1)
        batch_ix,start = nz[np.random.randint(nz.shape[0])]
        seq_hist = np.stack([z[batch_ix,start:start+seq_length] for z in past_stateseqs])
        axs[3].imshow(seq_hist, cmap=plt.cm.jet, aspect='auto', interpolation='nearest')
        axs[3].set_xlabel('Time (frames)')
        axs[3].set_ylabel('Iterations')
        axs[3].set_title('Stateseq history')
        
    fig.suptitle(f'Iteration {iteration}')
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
    
    
    



def crowd_movie_tile(key, start, end, videos, centroids, headings, 
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
                dot_color=(255,255,255), window_size=112, 
                pre=30, post=60, dot_radius=4):
    
    """Generate a crowd movie

    Parameters
    ----------
    instances: list of tuples ``(key, start, end)``
        List of syllable instances to include in the crowd movie,
        where each instance is specified as a tuple with the video 
        name, start frame and end frame. The list must have length
        ``rows*cols``. The video names must also be keys in ``videos``.
        
    rows: int
        Number of rows in the crowd movie grid
        
    cols: int
        Number of columns in the crowd movie grid
        
    videos: dict
        Dictionary mapping video names to video readers. Frames from
        each reader should be accessible via __getitem__(int or slice).

    Returns
    -------

    """

    tiles = np.stack([
        crowd_movie_tile(
            key, start, end, videos, centroids, headings, 
            dot_color=dot_color, window_size=window_size,
            pre=pre, post=post, dot_radius=dot_radius
        ) for key, start, end in instances
    ]).reshape(rows, cols, post+pre, window_size, window_size, 3)
    return np.concatenate(np.concatenate(tiles,axis=2),axis=2)


    
def write_video_clip(frames, path, fps=30, quality=7):
            
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
    dot_color=(255,255,255), window_size=112, plot_keypoints=False, 
    use_reindexed=True, sampling_options={}, coordinates=None, 
    bodyparts=None, use_bodyparts=None, quality=7, **kwargs):
    
    assert (video_dir is not None) or (video_paths is not None), fill(
        'You must provide either ``video_dir`` or ``video_paths``')
            
    if plot_keypoints:
        raise NotImplementedError()
        assert coordinates is not None, fill(
            '``coordinates`` are required when ``plot_keypoints==True``')        
    
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
        video_paths = find_matching_videos(results.keys(), video_dir, as_dict=True)
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

        
        

def pad_limits(limits, left=0.1, right=0.1, top=0.1, bottom=0.1):
    
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
                      cmap='autumn', node_size=50, linewidth=3, 
                      fig_width=4, overlap=(0.2,0)):
    
    num_timesteps = Xs[0].shape[0]
    num_keypoints = Xs[0].shape[1]

    interval = int(np.floor(num_timesteps/10))
    plot_frames = np.arange(0,num_timesteps,interval)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0,1,num_keypoints))
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
        
    aspect = (ymax-ymin)/(xmax-xmin)
    fig.set_size_inches((fig_width*n_cols, fig_width*aspect*n_cols*1.1))
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
    n_neighbors=40, keypoint_colormap='autumn', fig_size=4,
    grid_cols=5, grid_margin=(-.2,-.2), plot_options={}, 
    sampling_options={}, **kwargs):

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
    plot_options = {**plot_options, 'cmap':keypoint_colormap}
        
    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration,
        min_frequency=min_frequency, min_instances=n_neighbors)
    
    sampled_instances = sample_instances(
        syllable_instances, n_neighbors, coordinates=coordinates, 
        centroids=centroids, headings=headings, n_neighbors=n_neighbors, 
        **sampling_options)

    trajectories = get_trajectories(
        sampled_instances, coordinates, pre=pre, post=post, 
        centroids=centroids, headings=headings)

    if skeleton is None: edges = []
    else: edges = get_edges(use_bodyparts, skeleton)

    syllables = sorted(trajectories.keys())
    titles = [f'Syllable {syllable}' for syllable in syllables]
    Xs = np.array([trajectories[syllable] for syllable in syllables]).mean(1)
    
    lims = np.stack([Xs.min((0,1,2)),Xs.max((0,1,2))])
    lims = pad_limits(lims, left=0.1, right=0.1, top=0.2, bottom=0.2)

        
    if Xs.shape[-1]==2:
        
        # individual plots
        for title,X in zip(titles,Xs):
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
