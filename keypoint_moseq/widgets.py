"""
This module contains the widget components in the analysis and visualiation pipeline.
"""

# group setting widget imports
import os
import qgrid
import pandas as pd
import yaml
import ipywidgets as widgets
from IPython.display import display, clear_output

# video viewer widget additional imports
import io
import imageio
import base64
from glob import glob
from bokeh.io import show
from bokeh.models import Div, CustomJS, Slider


class GroupSettingWidgets:
    """The group setting widget for setting group names in the index file.
    """

    def __init__(self, index_filepath):
        """
        Initialize all the Group Setting widgets, parses the index yaml file into a pandas DataFrame that is compatible to be displayed using QGrid.

        Args:
        index_filepath (str): Path to index file (moseq2-index.yaml) containing session metadata and grouping info.
        """

        self.index_filepath = index_filepath
        style = {'description_width': 'initial',
                 'display': 'flex-grow', 'align_items': 'stretch'}

        self.col_opts = {
            'editable': False,
            'toolTip': "Not editable"
        }

        self.col_defs = {
            'group': {
                'editable': True,
                'toolTip': 'editable'
            }
        }

        self.clear_button = widgets.Button(
            description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', style=style,
                                        description='New Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group Name', style=style,
                                          disabled=False, tooltip='Set Group')
        self.update_index_button = widgets.Button(description='Update Index File', style=style,
                                                  disabled=False, tooltip='Save Parameters')

        self.group_set = widgets.HBox(
            [self.group_input, self.save_button, self.update_index_button])

        self.index_dict, self.df = self.index_to_dataframe(self.index_filepath)
        self.qgrid_widget = qgrid.show_grid(self.df[['group', 'uuid', 'filename']],
                                            column_options=self.col_opts,
                                            column_definitions=self.col_defs,
                                            show_toolbar=False)

        qgrid.set_grid_option('forceFitColumns', False)
        qgrid.set_grid_option('enableColumnReorder', True)
        qgrid.set_grid_option('highlightSelectedRow', True)
        qgrid.set_grid_option('highlightSelectedCell', False)

        # Add callback functions
        self.clear_button.on_click(self.clear_clicked)
        self.update_index_button.on_click(self.update_clicked)
        self.save_button.on_click(self.update_table)

    def index_to_dataframe(self, index_filepath):
        """parse index file to a dataframe

        Parameters
        ----------
        index_filepath : str
            path to the index file

        Returns
        -------
        index_data : dict
            the dictionary containing the index data
        df : pandas.DataFrame
            the dataframe containing the index data
        """

        # load index data
        with open(index_filepath, 'r') as f:
            index_data = yaml.safe_load(f)

        # process index data into dataframe
        df = pd.DataFrame(index_data['files'])

        return index_data, df

    def update_table(self, b=None):
        """
        Update table upon "Set Button" click

        Args:
        b (button click)
        """

        self.update_index_button.button_style = 'info'
        self.update_index_button.icon = 'none'

        selected_rows = self.qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            self.qgrid_widget.edit_cell(i, 'group', self.group_input.value)

    def update_clicked(self, b=None):
        """
        Update the index file with the current table state upon Save button click.

        Args:
        b (button click)
        """

        files = self.index_dict['files']

        latest_df = self.qgrid_widget.get_changed_df()
        self.df.update(latest_df)

        updated_index = {'files': list(
            self.df.to_dict(orient='index').values())}

        with open(self.index_filepath, 'w') as f:
            yaml.safe_dump(updated_index, f, default_flow_style=False)

        self.update_index_button.button_style = 'success'
        self.update_index_button.icon = 'check'

    def clear_clicked(self, b=None):
        """
        Clear the display.

        Args:
        b (ipywidgets.Button click): callback from button when user clicks the button.
        Returns:
        """
        clear_output()


class InteractiveVideoViewer:
    """The interactive video viewer widget for viewing grid movies or crowd movies.
    """

    def __init__(self, syll_vid_dir):
        """
        initialize the video viewer widget.

        Args:
        syll_vid_dir (str): Path to base directory containing all syllable movies to show
        """

        self.sess_select = widgets.Dropdown(options=self.create_syllable_path_dict(syll_vid_dir),
                                            description='Syllable:', disabled=False, continuous_update=True)

        self.clear_button = widgets.Button(
            description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.clear_button.on_click(self.clear_on_click)

    def create_syllable_path_dict(self, syll_vid_dir):
        """
        Create a dictionary of all syllable videos in the syllable video directory.

        Args:
        syll_vid_dir (str): Path to syllable video directory.
        """

        syll_vid_dict = {}
        try:
            files = sorted(glob(os.path.join(syll_vid_dir, '*.mp4')),
                           key=lambda x: int(os.path.basename(x).split('.')[0][8:]))
        except ValueError:
            print(
                'Syllable name not in the format of "syllable#.mp4", syllable videos will not be sorted.')
            files = glob(os.path.join(syll_vid_dir, '*.mp4'))

        for file in files:
            file_name = os.path.basename(file)
            syll_vid_dict[file_name] = file

        return syll_vid_dict

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()

    def get_video(self, input_file):
        """
        Returns a div containing a video object to display.

        Args:
        input_file (str): Path to session extraction video to view.
        """

        # get video dimensions
        vid = imageio.get_reader(input_file, 'ffmpeg')
        video_dims = vid.get_meta_data()['size']

        # input_file goes through encode and decode so it won't carry semantic meanings anymore
        file_name = input_file

        # Open videos in encoded urls
        # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
        vid = io.open(input_file, 'r+b').read()
        encoded = base64.b64encode(vid)
        input_file = encoded.decode('ascii')

        video_div = f"""
                        <h2>{file_name}</h2>
                        <video
                            src="data:video/mp4;base64, {input_file}"; alt="data:video/mp4;base64, {input_file}"; id="preview";
                            height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                            style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                        <script>
                            document.querySelector('video').playbackRate = 0.1;
                        </script>
                     """

        div = Div(text=video_div, style={
                  'width': '100%', 'align-items': 'center', 'display': 'contents'})

        slider = Slider(start=0, end=4, value=1, step=0.1,
                        format="0[.]00", title=f"Playback Speed")

        callback = CustomJS(
            args=dict(slider=slider),
            code="""
                    document.querySelector('video').playbackRate = slider.value;
                 """
        )

        slider.js_on_change('value', callback)
        show(slider)
        show(div)


def show_trajectory_gif(progress_paths):
    """show trajectory gif for syllable labeling

    Parameters
    ----------
    progress_paths : dict
        dictionary of paths and filenames for progress tracking
    """

    trajectory_gif = os.path.join(
        progress_paths['trajectory_plot_dir'], 'all_trajectories.gif')

    if os.path.exists(trajectory_gif):
        with open(trajectory_gif, 'rb') as file:
            image = file.read()
        out = widgets.Image(value=image, format='gif')
        display(out)
    else:
        print('All trajectory gif not found. Please generate trajecotry gif first.')
