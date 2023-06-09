import os
import yaml
import numpy as np
from bokeh.io import output_notebook, show
from glob import glob
import ipywidgets as widgets
from IPython.display import display
from keypoint_moseq.widgets import GroupSettingWidgets, SyllableLabeler
from keypoint_moseq.io import load_results, load_deeplabcut_results, load_sleap_results, load_config
from keypoint_moseq.analysis import generate_index
from keypoint_moseq.viz import generate_grid_movies, generate_crowd_movies
output_notebook()

def interactive_group_setting(project_dir, model_dirname):
    """start the interactive group setting widget

    Parameters
    ----------
    project_dir : str
        the path to the project directory
    model_dirname : str
        the name of the model directory

    """

    index_filepath = os.path.join(project_dir, 'index.yaml')

    if os.path.exists(index_filepath):
        with open(index_filepath, 'r') as f:
            index_data = yaml.safe_load(f)
    else:
        generate_index(project_dir, model_dirname, index_filepath)

    # display the widget
    index_grid = GroupSettingWidgets(index_filepath)
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)
    return index_filepath


def label_syllables(project_dir, model_dirname, video_dir=None, keypoint_data_type='deeplabcut', movie_type='grid'):
    """label syllables in the syllable grid movie

    Parameters
    ----------
    progress_paths : dict
        the dictionary containing path names in the analysis process
    """

    output_notebook()

    config_data = load_config(project_dir)
    # load coordinates
    # if video_dir not specified, find it in the config file
    if video_dir is None:
        video_dir = config_data.get('video_dir', None)
    # if video_dir is still None, raise an error
    if video_dir is None:
        raise Exception('Unable to find video directory. Please specify video directory.')
    grid_movies = glob(os.path.join(project_dir, model_dirname, 'grid_movies', '*.mp4'))
    crowd_movies = glob(os.path.join(project_dir, model_dirname, 'crowd_movies', '*.mp4'))
    if len(grid_movies)==0 or len(crowd_movies) == 0:
        if keypoint_data_type == 'deeplabcut':
            coordinates, _, _ = load_deeplabcut_results(video_dir)
        elif keypoint_data_type == 'sleap':
            coordinates, _, _ = load_sleap_results(video_dir)
        else:
            raise NotImplementedError('Input type not supported.')

        # check if movies are generated
        if len(grid_movies)==0:
            print('No grid movies found in the directory. Generating grid movies')
            generate_grid_movies(name=model_dirname, project_dir=project_dir, coordinates=coordinates, **config_data)
            # record the movie paths
            grid_movies=glob(os.path.join(project_dir, model_dirname, 'grid_movies', '*.mp4'))
        
        
        if len(crowd_movies)==0:
            print('No crowd movies found in the directory. Generating crowd movies')
            generate_crowd_movies(name=model_dirname, project_dir=project_dir, coordinates=coordinates, **config_data)
            # record the movie paths
            crowd_movies=glob(os.path.join(project_dir, model_dirname, 'crowd_movies', '*.mp4'))

    # construct the syllable info path
    syll_info_path = os.path.join(project_dir, model_dirname, "syll_info.yaml")

    # generate a new syll_info yaml file
    if not os.path.exists(syll_info_path):
        # parse model results
        model_results = load_results(project_dir, model_dirname)
        unique_sylls = np.unique(np.concatenate([file['syllables_reindexed'] for file in model_results.values()]))
        # construct the syllable dictionary
        syll_dict = {int(i): {'label': '', 'desc': '', 'movie_path': [], 'group_info': {}} for i in unique_sylls}
        # record the movie paths
        for movie_path in grid_movies:
            syll_index = int(os.path.splitext(os.path.basename(movie_path))[0][8:])
            syll_dict[syll_index]['movie_path'].append(movie_path)
        for movie_path in crowd_movies:
            syll_index = int(os.path.splitext(os.path.basename(movie_path))[0][8:])
            syll_dict[syll_index]['movie_path'].append(movie_path)

        # write to file
        print(syll_info_path)
        with open(syll_info_path, 'w') as file:
            yaml.safe_dump(syll_dict, file, default_flow_style=False)

    # construct the index path
    index_path = os.path.join(project_dir, "index.yaml")
    # create index.yaml if it does not exist
    if not os.path.exists(index_path):
        print('index.yaml does not exist, creating one...')
        generate_index(project_dir, model_dirname, index_path)
    labeler = SyllableLabeler(project_dir, model_dirname, index_path, syll_info_path, video_dir, keypoint_data_type, movie_type)
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.clear_button, labeler.syll_select, output)
