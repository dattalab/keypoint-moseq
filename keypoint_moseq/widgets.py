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

# syllable labeler widget and controller additional imports
import re
import numpy as np
import pandas as pd
from copy import deepcopy
from bokeh.io import show
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
from os.path import exists
from bokeh.models import Div, CustomJS, Slider
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText
from keypoint_moseq.analysis import compute_moseq_df, compute_stats_df


def read_yaml(yaml_file):
    """read yaml file into dictionary

    Parameters
    ----------
    yaml_file : str
        path to yaml file

    Returns
    -------
    dict
        dictionary of yaml file data
    """

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


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


class SyllableLabelerWidgets:

    def __init__(self):
        """
        launch the widget for labelling syllables with name and descriptions using the crowd movies.
        """
        self.clear_button = widgets.Button(
            description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.syll_select = widgets.Dropdown(
            options={}, description='Syllable #:', disabled=False)

        # labels
        self.cm_lbl = PreText(text="Syllable Movie")  # current movie number

        self.syll_lbl = widgets.Label(
            value="Syllable Name")  # name user prompt label
        self.desc_lbl = widgets.Label(
            value="Short Description")  # description label

        self.syll_info_lbl = widgets.Label(value="Syllable Info", font_size=24)

        self.syll_usage_value_lbl = widgets.Label(value="")
        self.syll_speed_value_lbl = widgets.Label(value="")
        self.syll_duration_value_lbl = widgets.Label(value="")

        # text input widgets
        self.lbl_name_input = widgets.Text(value='',
                                           placeholder='Syllable Name',
                                           tooltip='2 word name for syllable')

        self.desc_input = widgets.Text(value='',
                                       placeholder='Short description of behavior',
                                       tooltip='Describe the behavior.',
                                       disabled=False)

        # buttons
        self.prev_button = widgets.Button(description='Prev', disabled=False, tooltip='Previous Syllable', layout=widgets.Layout(
            flex='2 1 0', width='auto', height='40px'))
        self.set_button = widgets.Button(description='Save Setting', disabled=False, tooltip='Save current inputs.',
                                         button_style='primary', layout=widgets.Layout(flex='3 1 0', width='auto', height='40px'))
        self.next_button = widgets.Button(description='Next', disabled=False, tooltip='Next Syllable', layout=widgets.Layout(
            flex='2 1 0', width='auto', height='40px'))

        # Box Layouts
        self.label_layout = widgets.Layout(flex_flow='column', height='100%')

        self.ui_layout = widgets.Layout(flex_flow='row', width='auto')

        self.data_layout = widgets.Layout(flex_flow='row', padding='top',
                                          align_content='center', justify_content='space-around',
                                          width='100%')

        self.data_col_layout = widgets.Layout(flex_flow='column',
                                              align_items='center',
                                              align_content='center',
                                              justify_content='space-around',
                                              width='100%')

        self.center_layout = widgets.Layout(justify_content='space-around',
                                            align_items='center')

        # label box
        self.lbl_box = VBox([self.syll_lbl, self.desc_lbl],
                            layout=self.label_layout)

        # input box
        self.input_box = VBox(
            [self.lbl_name_input, self.desc_input], layout=self.label_layout)

        # syllable info box
        self.info_boxes = VBox([self.syll_info_lbl], layout=self.center_layout)

        self.data_box = VBox([HBox([self.lbl_box, self.input_box], layout=self.data_layout), self.info_boxes],
                             layout=self.data_col_layout)

        # button box
        self.button_box = HBox(
            [self.prev_button, self.set_button, self.next_button], layout=self.ui_layout)

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()
        del self

    def on_next(self, event=None):
        """
        trigger an view update when the user clicks the "Next" button.

        Args:
        event (ipywidgets.ButtonClick): User clicks next button.
        """

        # Updating dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index < len(self.syll_select.options) - 1:
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_prev(self, event=None):
        """
        trigger an view update when the user clicks the "Previous" button.

        Args:
        event (ipywidgets.ButtonClick): User clicks 'previous' button.
        """

        # Update syllable information dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = len(self.syll_select.options) - 1

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_set(self, event=None):
        """
        save the dict to syllable information file.

        Args:
        event (ipywidgets.ButtonClick): User clicks the 'Save' button.
        """

        # Update dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        self.write_syll_info(curr_syll=self.syll_select.index)

        # Update button style
        self.set_button.button_style = 'success'


class SyllableLabeler(SyllableLabelerWidgets):

    def __init__(self, base_dir, model_name, index_file, movie_type, syll_info_path):
        """
        Initialize syllable labeler widget with class context parameters, and create the syllable information dict.

        Args:
        model_fit (dict): Loaded trained model dict.
        index_file (str): Path to saved index file.
        max_sylls (int): Maximum number of syllables to preview and label.
        select_median_duration_instances (bool): boolean flag to select examples with syallable duration closer to median.
        save_path (str): Path to save syllable label information dictionary.
        """

        super().__init__()

        if movie_type == 'grid':
            movie_dir = os.path.join(base_dir, model_name, 'grid_movies')
        else:
            movie_dir = os.path.join(base_dir, model_name, 'crowd_movies')

        try:
            input_file = glob(os.path.join(movie_dir, '*.mp4'))[0]
            vid = imageio.get_reader(input_file, 'ffmpeg')
            video_dims = vid.get_meta_data()['size']
        except IndexError:
            print('No syllable movies found in the directory.')

        self.base_dir = base_dir
        self.model_name = model_name
        self.index_file = index_file
        self.sorted_index = read_yaml(yaml_file=index_file)
        self.movie_type = movie_type
        self.syll_info_path = syll_info_path
        self.video_dims = video_dims

        # check if syllable info file exists
        if os.path.exists(syll_info_path):
            self.syll_info = read_yaml(syll_info_path)
        else:
            self.syll_info = self._initialize_syll_info_dict(self.model_name)
            # Write to file
            with open(self.syll_info_path, 'w') as f:
                yaml.safe_dump(self.syll_info, f, default_flow_style=False)

        # Initialize button callbacks
        self.next_button.on_click(self.on_next)
        self.prev_button.on_click(self.on_prev)
        self.set_button.on_click(self.on_set)
        self.clear_button.on_click(self.clear_on_click)

        # generate by group syllable statistics dictionary
        self.get_group_df()

        # Get dropdown options with labels
        self.option_dict = {f'{i} - {x["label"]}': self.syll_info[i]
                            for i, x in enumerate(self.syll_info.values())}

        # Set the syllable dropdown options
        self.syll_select.options = self.option_dict

    def _initialize_syll_info_dict(self, movie_dir):
        grid_movie_files = sorted(glob(os.path.join(self.base_dir, self.model_name, 'grid_movies',
                                  '*.mp4')), key=lambda x: int(os.path.basename(x).split('.')[0][8:]))
        crowd_movie_files = sorted(glob(os.path.join(self.base_dir, self.model_name,
                                   'crowd_movies', '*.mp4')), key=lambda x: int(os.path.basename(x).split('.')[0][8:]))
        return {i: {'label': '', 'desc': '', 'movie_path': [grid_movie_files[i], crowd_movie_files[i]], 'group_info': {}} for i in range(len(grid_movie_files))}

    def write_syll_info(self, curr_syll=None):
        """
        Write current syllable info data to a YAML file.
        """

        # Dropping group info from dict
        tmp = deepcopy(self.syll_info)
        for v in tmp.values():
            v['group_info'] = {}

        # Write to file
        with open(self.syll_info_path, 'w') as f:
            yaml.safe_dump(tmp, f, default_flow_style=False)

        if curr_syll is not None:
            # Update the syllable dropdown options
            self.option_dict = {f'{i} - {x["label"]}': self.syll_info[i]
                                for i, x in enumerate(self.syll_info.values())}

            self.syll_select._initializing_traits_ = True
            self.syll_select.options = self.option_dict
            self.syll_select.index = curr_syll
            self.syll_select._initializing_traits_ = False

    def get_group_df(self):
        """
        Populate syllable information dict with usage and scalar information.
        """
        print('Computing Syllable Statistics...')
        moseq_df = compute_moseq_df(
            self.base_dir, self.model_name, self.index_file)
        stats_df = compute_stats_df(moseq_df, groupby=['group'])[
            ['group', 'syllable', 'frequency', 'duration', 'heading_mean', 'velocity_px_s_mean']]

        # Get all unique groups in df
        self.groups = stats_df.group.unique()

        self.group_syll_info = deepcopy(self.syll_info)
        for syll_key, syll_info in self.group_syll_info.items():
            for group in self.groups:
                # initialize group info dict
                syll_info['group_info'][group] = {}
                syll_info['group_info'][group]['frequency'] = stats_df[(
                    (stats_df.group == group) & (stats_df.syllable == syll_key))]['frequency'].values[0]
                syll_info['group_info'][group]['duration (s)'] = stats_df[(
                    (stats_df.group == group) & (stats_df.syllable == syll_key))]['duration'].values[0]
                syll_info['group_info'][group]['velocity (pixel/s)'] = stats_df[(
                    (stats_df.group == group) & (stats_df.syllable == syll_key))]['velocity_px_s_mean'].values[0]

    def set_group_info_widgets(self, group_info):
        """
        read the syllable information into a pandas DataFrame and display it as a table.

        Args:
        group_info (dict): Dictionary of grouped current syllable information
        """

        full_df = pd.DataFrame(group_info)
        columns = full_df.columns

        output_tables = []
        if len(self.groups) < 4:
            # if there are less than 4 groups, plot the table in one row
            output_tables = [Div(text=full_df.to_html())]
        else:
            # plot 4 groups per row to avoid table being cut off by movie
            n_rows = int(len(columns) / 4)
            row_cols = np.array_split(columns, n_rows)

            for i in range(len(row_cols)):
                row_df = full_df[row_cols[i]]
                output_tables += [Div(text=row_df.to_html())]

        ipy_output = widgets.Output()
        with ipy_output:
            for ot in output_tables:
                show(ot)

        self.info_boxes.children = [self.syll_info_lbl, ipy_output, ]

    def interactive_syllable_labeler(self, syllables):
        """
        create a Bokeh Div object to display the current video path.

        Args:
        syllables (int or ipywidgets.DropDownMenu): Current syllable to label
        """

        self.set_button.button_style = 'primary'

        # Set current widget values
        if len(syllables['label']) > 0:
            self.lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            self.desc_input.value = syllables['desc']

        # Update label
        self.cm_lbl.text = f'Crowd Movie {self.syll_select.index + 1}/{len(self.syll_select.options)}'

        # Update scalar values
        self.set_group_info_widgets(
            self.group_syll_info[self.syll_select.index]['group_info'])

        # Get current movie path
        if self.movie_type == 'grid':
            cm_path = syllables['movie_path'][0]
        else:
            cm_path = syllables['movie_path'][1]

        video_dims = self.video_dims

        # open the video and encode to be displayed in jupyter notebook
        # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
        video = io.open(cm_path, 'r+b').read()
        encoded = base64.b64encode(video)

        # Create syllable crowd movie HTML div to embed
        video_div = f"""
                        <h2>{self.syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="data:video/mp4;base64,{encoded.decode("ascii")}"; alt="data:{cm_path}"; height="{video_dims[1]}"; width="{video_dims[0]}"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    """

        # Create embedded HTML Div and view layout
        div = Div(text=video_div, style={'width': '100%'})

        slider = Slider(start=0, end=2, value=1, step=0.1, width=video_dims[0]-50,
                        format="0[.]00", title=f"Playback Speed")

        callback = CustomJS(
            args=dict(slider=slider),
            code="""
                    document.querySelector('video').playbackRate = slider.value;
                 """
        )

        slider.js_on_change('value', callback)

        layout = column([div, self.cm_lbl, slider])

        # Insert Bokeh div into ipywidgets Output widget to display
        vid_out = widgets.Output(layout=widgets.Layout(display='inline-block'))
        with vid_out:
            show(layout)

        # Create grid layout to display all the widgets
        grid = widgets.AppLayout(left_sidebar=vid_out,
                                 right_sidebar=self.data_box,
                                 pane_widths=[3, 0, 3])

        # Display all widgets
        display(grid, self.button_box)