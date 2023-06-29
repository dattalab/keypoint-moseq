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
from bokeh.io import show
from bokeh.models import Div, CustomJS, Slider

# syllable labeler widget and controller additional imports
import numpy as np
import pandas as pd
from copy import deepcopy
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Div, CustomJS, Slider
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText
from keypoint_moseq.io import load_results
from keypoint_moseq.util import filter_angle, get_frequencies


def compute_moseq_df(base_dir, model_name, *, fps=30, smooth_heading=True):
    """compute moseq dataframe from results dict that contains all kinematic values by frame
    Parameters
    ----------
    base_dir : str
        the path to the project directory
    model_name : str
        the name of the model directory
    results_dict : dict
        dictionary of results from model fitting
    use_bodyparts : bool
        boolean flag whether to include data for bodyparts
    smooth_heading : bool, optional
        boolean flag whether smooth the computed heading, by default True

    Returns
    -------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    """

    # load model results
    results_dict = load_results(base_dir, model_name)

    # load index file
    index_filepath = os.path.join(base_dir, 'index.yaml')
    if os.path.exists(index_filepath):
        with open(index_filepath, 'r') as f:
            index_data = yaml.safe_load(f)

        # create a file dictionary for each session
        file_info = {}
        for session in index_data['files']:
            file_info[session['name']] = {'group': session['group']}
    else:
        print('index.yaml not found, if you want to include group information for each video, please run the Assign Groups widget first')

    session_name = []
    centroid = []
    velocity = []
    heading = []
    angular_velocity = []
    syllables = []
    frame_index = []
    s_group = []

    for k, v in results_dict.items():
        n_frame = v['centroid'].shape[0]
        session_name.append([str(k)] * n_frame)
        centroid.append(v['centroid'])
        # velocity is pixel per second
        velocity.append(np.concatenate(
            ([0], np.sqrt(np.square(np.diff(v['centroid'], axis=0)).sum(axis=1)) * fps)))

        if file_info is not None:
            # find the group for each session from index data
            s_group.append([file_info[k]['group']]*n_frame)
        else:
            # no index data
            s_group.append(['default']*n_frame)
        frame_index.append(np.arange(n_frame))

        if smooth_heading:
            session_heading = filter_angle(v['heading'])
        else:
            session_heading = v['heading']

        # heading in radian
        heading.append(session_heading)
        # compute angular velocity (radian per second)
        angular_velocity.append(np.concatenate(
            ([0], np.diff(session_heading) * fps)))

        # add syllable data
        syllables.append(v['syllable'])

    # construct dataframe
    moseq_df = pd.DataFrame(np.concatenate(
        session_name), columns=['name'])
    moseq_df = pd.concat([moseq_df, pd.DataFrame(np.concatenate(centroid), columns=[
                         'centroid_x', 'centroid_y'])], axis=1)
    moseq_df['heading'] = np.concatenate(heading)
    moseq_df['angular_velocity'] = np.concatenate(angular_velocity)
    moseq_df['velocity_px_s'] = np.concatenate(velocity)
    moseq_df['syllable'] = np.concatenate(syllables)
    moseq_df['frame_index'] = np.concatenate(frame_index)
    moseq_df['group'] = np.concatenate(s_group)

    # compute syllable onset
    change = np.diff(moseq_df['syllable']) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))

    onset = np.full(moseq_df.shape[0], False)
    onset[indices] = True
    moseq_df['onset'] = onset
    return moseq_df


def compute_stats_df(base_dir, model_name, moseq_df, min_frequency=0.005, groupby=['group', 'name'], fps=30, normalize=True, **kwargs):
    """summary statistics for syllable frequencies and kinematic values
    Parameters
    ----------
    moseq_df : pandas.DataFrame
        the dataframe that contains kinematic data for each frame
    threshold : float, optional
        usge threshold for the syllable to be included, by default 0.005
    groupby : list, optional
        the list of column names to group by, by default ['group', 'name']
    fps : int, optional
        frame per second information of the recording, by default 30
    normalize : bool, optional
        boolean falg whether to normalize by counts, by default True

    Returns
    -------
    stats_df : pandas.DataFrame
        the summary statistics dataframe for syllable frequencies and kinematic values
    """
    # compute runlength encoding for syllables
    
    # load model results
    results_dict = load_results(base_dir, model_name)
    syllables = {k:res['syllable'] for k,res in results_dict.items()}
    # frequencies is array of frequencies for sorted syllables [syll_0, syll_1...]
    frequencies=get_frequencies(syllables)
    syll_include=np.where(frequencies>min_frequency)[0]

    # add group information
    # load index file
    index_filepath = os.path.join(base_dir, 'index.yaml')
    if os.path.exists(index_filepath):
        with open(index_filepath, 'r') as f:
            index_data = yaml.safe_load(f)

        # create a file dictionary for each session
        file_info = {}
        for session in index_data['files']:
            file_info[session['name']] = {'group': session['group']}
    else:
        print('index.yaml not found, if you want to include group information for each video, please run the Assign Groups widget first')

    # construct frequency dataframe
    frequency_df = []
    for k, v in results_dict.items():
        syll_freq=get_frequencies(v['syllable'])
        df=pd.DataFrame({'name': k,
                         'group': file_info[k]['group'],
                         'syllable': np.arange(len(syll_freq)),
                         'frequency': syll_freq})
        frequency_df.append(df)
    frequency_df = pd.concat(frequency_df)
    if 'name' not in groupby:
        frequency_df.drop(columns=['name'], inplace=True)
    frequency_df=frequency_df.groupby(groupby + ['syllable']).mean().reset_index()

    # filter out syllables that are used less than threshold in all sessions
    filtered_df = moseq_df[moseq_df['syllable'].isin(syll_include)].copy()

    # TODO: hard-coded heading for now, could add other scalars
    features = filtered_df.groupby(
        groupby + ['syllable'])[['heading', 'angular_velocity', 'velocity_px_s']].agg(['mean', 'std', 'min', 'max'])

    features.columns = ['_'.join(col).strip()
                        for col in features.columns.values]
    features.reset_index(inplace=True)

    # get durations
    trials = filtered_df['onset'].cumsum()
    trials.name = 'trials'
    durations = filtered_df.groupby(
        groupby + ['syllable'] + [trials])['onset'].count()
    # average duration in seconds
    durations = durations.groupby(groupby + ['syllable']).mean() / fps
    durations.name = 'duration'
    # only keep the columns we need
    durations = durations.fillna(0).reset_index()[groupby + ['syllable', 'duration']]
    
    stats_df = pd.merge(features, frequency_df, on=groupby+['syllable'])
    stats_df = pd.merge(stats_df, durations, on=groupby+['syllable'])
    return stats_df


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


def show_trajectory_gif(project_dir, model_dirname):
    """show trajectory gif for syllable labeling

    Parameters
    ----------
    progress_paths : dict
        dictionary of paths and filenames for progress tracking
    """
    trajectory_gifs_path = os.path.join(
        project_dir, model_dirname, 'trajectory_plots', 'all_trajectories.gif')

    assert os.path.exists(trajectory_gifs_path), (
        f'Trajectory plots not found at {trajectory_gifs_path}. '
        'See documentation for generating trajectory plots: '
        'https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html#visualization')

    with open(trajectory_gifs_path, 'rb') as file:
        image = file.read()
    out = widgets.Image(value=image, format='gif')
    display(out)


class GroupSettingWidgets:
    """The group setting widget for setting group names in the index file.
    """

    def __init__(self, index_filepath):
        """Initialize the group setting widget

        Parameters
        ----------
        index_filepath : str
            path to the index file
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
        self.qgrid_widget = qgrid.show_grid(self.df[['group', 'name']],
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
        """Update table upon "Set Button" click

        Parameters
        ----------
        b : button click, optional
            button click to udpate table, by default None
        """

        self.update_index_button.button_style = 'info'
        self.update_index_button.icon = 'none'

        selected_rows = self.qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            self.qgrid_widget.edit_cell(i, 'group', self.group_input.value)

    def update_clicked(self, b=None):
        """Update the index file with the current table state upon Save button click.

         Parameters
        ----------
        b : button click, optional
            button click to update and save the table, by default None
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
        """Clear the display.

        Parameters
        ----------
        b : button click, optional
            callback from button when user clicks the button, by default None
        """

        clear_output()


class SyllableLabelerWidgets:
    """The syllable labeler widgets for labeling syllables.
    """

    def __init__(self):
        """Initialize the syllable labeler widgets.
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
                                           tooltip='name for the syllable')

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

        Parameters
        ----------
        b: button click, optional
            Button click to clear the javascript output
        """

        clear_output()
        del self

    def on_next(self, event=None):
        """trigger an view update when the user clicks the "Next" button.

        Parameters
        ----------
        event: ipywidgets.ButtonClick
            User clicks next button.
        """

        # Updating dict
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index < len(self.syll_select.options) - 1:
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[self.syll_list[self.syll_select.index]]['label']
        self.desc_input.value = self.syll_info[self.syll_list[self.syll_select.index]]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_prev(self, event=None):
        """trigger an view update when the user clicks the "Previous" button.

        Parameters
        ----------
        event: ipywidgets.ButtonClick
            User clicks previous button.
        """

        # Update syllable information dict
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = len(self.syll_select.options) - 1

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[self.syll_list[self.syll_select.index]]['label']
        self.desc_input.value = self.syll_info[self.syll_list[self.syll_select.index]]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_set(self, event=None):
        """save the dict to syllable information file.
        Parameters
        ----------
        event: ipywidgets.ButtonClick
            User clicks save button.
        """

        # Update dict
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_list[self.syll_select.index]
                       ]['desc'] = self.desc_input.value

        self.write_syll_info(curr_syll=self.syll_select.index)

        # Update button style
        self.set_button.button_style = 'success'


class SyllableLabeler(SyllableLabelerWidgets):
    """Syllable Labeler control component.
    """

    def __init__(self, project_dir, model_dirname, moseq_df, index_path, syll_info_path, movie_type):
        """Initialize the SyllableLabeler

        Parameters
        ----------
        base_dir: str
            Base directory for the model.
        model_name: str
            Name of the model.
        index_file: str
            Path to the index file.
        movie_type: str
            Type of movie to load.
        syll_info_path: str
            Path to the syllable information file.
        """

        super().__init__()

        self.base_dir = project_dir
        self.model_name = model_dirname
        self.index_file = index_path
        self.sorted_index = read_yaml(yaml_file=index_path)
        self.movie_type = movie_type
        self.syll_info_path = syll_info_path
        # read in syllable information file and subset only those with crowd movies
        temp_syll_info = read_yaml(syll_info_path)
        self.syll_info = {
            k: v for k, v in temp_syll_info.items() if len(v['movie_path']) == 2}
        self.syll_list = sorted(list(self.syll_info.keys()))

        # Initialize button callbacks
        self.next_button.on_click(self.on_next)
        self.prev_button.on_click(self.on_prev)
        self.set_button.on_click(self.on_set)
        self.clear_button.on_click(self.clear_on_click)

        # generate by group syllable statistics dictionary
        self.get_group_df(moseq_df)

        # Get dropdown options with labels
        self.option_dict = {f'{i} - {x["label"]}': self.syll_info[i]
                            for i, x in self.syll_info.items()}

        # Set the syllable dropdown options
        self.syll_select.options = self.option_dict

    def write_syll_info(self, curr_syll=None):
        """Write current syllable info data to a YAML file.

        Parameters
        ----------
        curr_syll : int
            Current syllable index.
        """

        # Dropping group info from dict
        tmp = deepcopy(self.syll_info)
        for v in tmp.values():
            v['group_info'] = {}

        # read in the syllable information file
        temp_syll_info = read_yaml(self.syll_info_path)
        for k, v in temp_syll_info.items():
            if k in tmp.keys():
                temp_syll_info[k] = tmp[k]

        # Write to file
        with open(self.syll_info_path, 'w') as f:
            yaml.safe_dump(temp_syll_info, f, default_flow_style=False)

        if curr_syll is not None:
            # Update the syllable dropdown options
            self.option_dict = {f'{i} - {x["label"]}': self.syll_info[i]
                                for i, x in self.syll_info.items()}

            self.syll_select._initializing_traits_ = True
            self.syll_select.options = self.option_dict
            self.syll_select.index = curr_syll
            self.syll_select._initializing_traits_ = False

    def get_group_df(self, moseq_df):
        """Populate syllable information dict with usage and scalar information.
        """
        stats_df = compute_stats_df(self.base_dir, self.model_name, moseq_df, groupby=['group'])[
            ['group', 'syllable', 'frequency', 'duration', 'heading_mean', 'velocity_px_s_mean']]

        # Get all unique groups in df
        self.groups = stats_df.group.unique()

        self.group_syll_info = deepcopy(self.syll_info)
        for syll_info in self.group_syll_info.values():
            for group in self.groups:
                # make sure dataframe exists
                if len(stats_df[stats_df.group == group]) > 0:
                    # initialize group info dict
                    syll_info['group_info'][group] = {}
                    syll_info['group_info'][group]['frequency'] = stats_df[stats_df.group == group]['frequency'].values[0]
                    syll_info['group_info'][group]['duration (s)'] = stats_df[stats_df.group == group]['duration'].values[0]
                    syll_info['group_info'][group]['velocity (pixel/s)'] = stats_df[stats_df.group == group]['velocity_px_s_mean'].values[0]

    def set_group_info_widgets(self, group_info):
        """Read the syllable information into a pandas DataFrame and display it as a table.

        Parameters
        ----------
        group_info : dict
            Dictionary of grouped current syllable information.
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

        self.info_boxes.children = [self.syll_info_lbl, ipy_output]

    def interactive_syllable_labeler(self, syllables):
        """Create a Bokeh Div object to display the current video path.

        Parameters
        ----------
        syllables : int or ipywidgets.DropDownMenu
            Dictionary of syllable information.
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
            self.group_syll_info[self.syll_list[self.syll_select.index]]['group_info'])

        # Get current movie path
        if self.movie_type == 'grid':
            cm_path = syllables['movie_path'][0]
        else:
            cm_path = syllables['movie_path'][1]

        # open the video and encode to be displayed in jupyter notebook
        # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
        video = io.open(cm_path, 'r+b').read()
        encoded = base64.b64encode(video)
        video_dims = imageio.get_reader(
            cm_path, 'ffmpeg').get_meta_data()['size']

        # Create syllable crowd movie HTML div to embed
        video_div = f"""
                        <h2>{self.syll_list[self.syll_select.index]}: {syllables['label']}</h2>
                        <video
                            src="data:video/mp4;base64,{encoded.decode("ascii")}"; alt="data:{cm_path}"; height="{video_dims[1]}"; width="{video_dims[0]}"; preload="true";
                            style="float: left"; type: "video/mp4"; margin: 0px 10px 10px 0px;
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

        # slider.js_on_change('value', callback)

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
