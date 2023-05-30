import os
import uuid
import yaml
from bokeh.io import output_notebook, show
import ipywidgets as widgets
from IPython.display import display
from keypoint_moseq.widgets import GroupSettingWidgets, InteractiveVideoViewer, SyllableLabeler
from keypoint_moseq.io import load_results
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
        # generate a new index file
        results_dict = load_results(
            project_dir=project_dir, name=model_dirname)
        files = []
        for session in results_dict.keys():
            file_dict = {'filename': session, 'group': 'default',
                         'uuid': str(uuid.uuid4())}
            files.append(file_dict)

        index_data = {'files': files}
        # write to file and progress_paths
        with open(index_filepath, 'w') as f:
            yaml.safe_dump(index_data, f, default_flow_style=False)

    # display the widget
    index_grid = GroupSettingWidgets(index_filepath)
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)
    return index_filepath

def view_syllable_movies(progress_paths, movie_type='grid'):
    """view the syllable grid movie or crowd movie

    Parameters
    ----------
    progress_paths : dict
        the dictionary containing path names in the analysis process
    type : str, optional
        the type of movie to view, by default 'grid'
    """

    output_notebook()
    if movie_type == 'grid':
        # show grid movies
        video_dir = os.path.join(progress_paths['model_dir'], 'grid_movies')
        viewer = InteractiveVideoViewer(syll_vid_dir=video_dir)
    else:
        # show crowd movies
        video_dir = os.path.join(progress_paths['model_dir'], 'crowd_movies')
        viewer = InteractiveVideoViewer(syll_vid_dir=video_dir)

    # Run interactive application
    selout = widgets.interactive_output(viewer.get_video,
                                        {'input_file': viewer.sess_select})
    display(viewer.clear_button, viewer.sess_select, selout)


def label_syllables(project_dir, model_dirname, movie_type='grid'):
    """label syllables in the syllable grid movie

    Parameters
    ----------
    progress_paths : dict
        the dictionary containing path names in the analysis process
    """

    output_notebook()

    # check if syll_info.yaml exists
    syll_info_path = os.path.join(project_dir, model_dirname, "syll_info.yaml")
    index_path = os.path.join(project_dir, "index.yaml")

    labeler = SyllableLabeler(project_dir, model_dirname, index_path, movie_type, syll_info_path)
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.clear_button, labeler.syll_select, output)

    return progress_paths
