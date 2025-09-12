import dash_bootstrap_components as dbc
import numpy as np
import socket
import logging
import dash_player as dp
import flask
import os
import matplotlib.pyplot as plt
from IPython.display import display
from dash.dash_table import DataTable
from dash import Dash, html, dcc, callback, Input, Output, State
from typing import Callable, Mapping, Any

def _group_syllable_differences_plot(
    centers: np.ndarray,
    errors: np.ndarray,
    significant: list[list[tuple[int, int]]],
    group_labels: list[str],
    syllables: list[str | int],
    y_axis_label: str
):
    # Parameter shape checks
    if centers.shape != errors.shape:
        raise ValueError(f"centers and errors must have the same shape, got {centers.shape} and {errors.shape}")
    
    if centers.shape[0] != len(syllables):
        raise ValueError(f"Number of rows in centers ({centers.shape[0]}) must match length of syllables ({len(syllables)})")
    
    if centers.shape[1] != len(group_labels):
        raise ValueError(f"Number of columns in centers ({centers.shape[1]}) must match length of group_labels ({len(group_labels)})")
    
    if len(significant) != len(syllables):
        raise ValueError(f"Length of significant ({len(significant)}) must match length of syllables ({len(syllables)})")
    
    # Validate group indices in significance tuples
    for syllable_idx, comparisons in enumerate(significant):
        for group1_idx, group2_idx in comparisons:
            if not (0 <= group1_idx < len(group_labels)):
                raise ValueError(f"Invalid group index {group1_idx} in syllable {syllable_idx} significance. Must be 0-{len(group_labels)-1}")
            if not (0 <= group2_idx < len(group_labels)):
                raise ValueError(f"Invalid group index {group2_idx} in syllable {syllable_idx} significance. Must be 0-{len(group_labels)-1}")
            if group1_idx == group2_idx:
                raise ValueError(f"Cannot compare group {group1_idx} with itself in syllable {syllable_idx} significance")
    
    # Create matplotlib plot with dynamic width based on number of bars
    # Give each bar 0.25 inches width
    fig_width = len(syllables) * len(group_labels) * 0.25
    fig_height = 6  # Standard height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Calculate x positions and bar width for each group
    num_groups = len(group_labels)
    x_base = np.arange(len(syllables))
    bar_width = 0.8 / num_groups
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
    
    # Plot each group as bars with error bars
    for i, group in enumerate(group_labels):
        x_offset = (i - num_groups/2 + 0.5) * bar_width
        x_pos = x_base + x_offset
        
        ax.bar(x_pos, centers[:, i], bar_width, yerr=errors[:, i], 
               color=colors[i], label=group, capsize=3)
    
    # Add significance asterisks with dedicated lanes for each group pair
    # First, find all unique group pairs that have significance
    used_group_pairs = set()
    for comparisons in significant:
        for group1_idx, group2_idx in comparisons:
            # Ensure consistent ordering (smaller index first)
            pair = (min(group1_idx, group2_idx), max(group1_idx, group2_idx))
            used_group_pairs.add(pair)
    
    max_plot_height = 0
    if used_group_pairs:
        # Sort pairs for consistent lane assignment
        sorted_pairs = sorted(used_group_pairs)
        pair_to_lane = {pair: idx for idx, pair in enumerate(sorted_pairs)}
        
        # Find the global maximum y value (highest error bar top across all data)
        global_max_y = np.max(centers + errors)
        
        # Calculate lane positions
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] > 0 else 1
        base_offset = 0.05 * y_range
        lane_spacing = 0.04 * y_range
        
        # Place asterisks for each significant comparison
        for syllable_idx, comparisons in enumerate(significant):
            if not comparisons:  # Skip syllables with no significant comparisons
                continue
            
            # Add asterisk for each comparison at its dedicated lane
            syllable_x_center = x_base[syllable_idx]
            
            for group1_idx, group2_idx in comparisons:
                # Get the lane for this group pair
                pair = (min(group1_idx, group2_idx), max(group1_idx, group2_idx))
                lane_idx = pair_to_lane[pair]
                
                # Calculate y position for this lane
                y_lane = global_max_y + base_offset + lane_idx * lane_spacing
                
                # Calculate average color of the two groups being compared
                color1 = colors[group1_idx]
                color2 = colors[group2_idx]
                avg_color = [(c1 + c2) / 2 for c1, c2 in zip(color1[:3], color2[:3])]  # Average RGB, ignore alpha
                
                # Add asterisk at syllable center with averaged group color
                ax.text(syllable_x_center, y_lane, '*', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color=avg_color)
                
                # Track maximum height used
                max_plot_height = max(max_plot_height, y_lane + lane_spacing/2)
    
    # Extend y-axis if needed to show all significance lanes
    if max_plot_height > 0:
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1], max_plot_height + base_offset))
    
    # Set up x-axis
    ax.set_xticks(x_base)
    ax.set_xticklabels(syllables)
    ax.set_ylabel(y_axis_label)
    
    # Main legend for groups
    main_legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create second legend for significance lanes if there are any
    if used_group_pairs:
        from matplotlib.lines import Line2D
        
        # Create custom legend elements for significance lanes
        significance_handles = []
        significance_labels = []
        
        # Reverse order so top legend entry matches top lane
        for pair in reversed(sorted_pairs):
            group1_idx, group2_idx = pair
            
            # Calculate the same averaged color
            color1 = colors[group1_idx]
            color2 = colors[group2_idx]
            avg_color = [(c1 + c2) / 2 for c1, c2 in zip(color1[:3], color2[:3])]
            
            # Create a custom line element with asterisk marker
            handle = Line2D([0], [0], marker='*', color='w', markerfacecolor=avg_color, 
                           markersize=12, markeredgecolor=avg_color, linestyle='None')
            significance_handles.append(handle)
            
            # Create label with group names
            label = f"{group_labels[group1_idx]}-{group_labels[group2_idx]}"
            significance_labels.append(label)
        
        # Add second legend below the first one
        significance_legend = ax.legend(significance_handles, significance_labels, 
                                      bbox_to_anchor=(1.05, 1-0.05*(len(group_labels)+1)), 
                                      loc='upper left', title='Significance')
        
        # Add the main legend back (matplotlib removes it when creating a new one)
        ax.add_artist(main_legend)
    
    return fig

def get_video_dimensions(video_path):
    """Get video dimensions using OpenCV or fallback methods"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return width, height
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback: return reasonable default dimensions
    return 800, 600

def _set_group_labels_widget(initial_group_labels: Mapping[str, str], save_group_labels: Callable[[dict[str, str]], None]) -> Dash:
    """Creates a widget for labeling each recording session with an experimental group label.

    Parameters
    ----------
    initial_group_labels: Mapping[str, str]
        A mapping from recording session names to experimental group labels

    save_group_labels: Callable[[dict[str, str]], None]
        Callback function that runs when 'Save Group Assignments' is clicked.
        The input parameter has the same structure as initial_group_labels:
        keys are recording session names and values are experimental group names.

    Returns
    -------
    Dash
        The configured Dash widget instance
    """
    logging.debug(f'Starting group labels widget with initial group labels: {initial_group_labels}')
    records = [
        {'recording-name': name, 'group-label': group} 
        for name, group in initial_group_labels.items()
    ]

    columns = [
        {'name': 'Recording Name', 'id': 'recording-name', 'clearable': False, 'editable': False},
        {'name': 'Group Label', 'id': 'group-label', 'clearable': True}
    ]

    table = DataTable(
        id='group-labels-table',
        data=records,
        columns=columns,
        editable=True,
        style_cell_conditional=[
            {'if': {'column_id': 'recording-name'}, 'width': '10%'},
            {'if': {'column_id': 'recording-name'}, 'textAlign': 'left'}
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': '1px solid #ddd',
            'padding': '12px'
        },
        style_cell={
            'padding': '10px',
            'border': '1px solid #ddd',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_table={
            'overflowX': 'auto',
            'boxShadow': '0 0 20px rgba(0, 0, 0, 0.1)',
            'borderRadius': '8px'
        }
    )

    save_button = dbc.Button('Save Group Assignments', id='save-button')

    default_user_message = "Click to save group assignments"
    user_message = html.Div(id='user-message', children=default_user_message)

    layout = dbc.Row(
        [dbc.Col(table), dbc.Col([user_message, save_button])],
        style={'margin': '20px'})

    @callback(
        Output('user-message', 'children'),
        Input('save-button', 'n_clicks'),
        State('group-labels-table', 'data'),
        prevent_initial_call=True
    )
    def _save_group_labels(_, table_data):
        group_labels = {
            row['recording-name']: row.get('group-label', '') or ''
            for row in table_data if row['recording-name']
        }

        logging.debug(f'Saving group labels {group_labels}.')
        save_group_labels(group_labels)
        return 'Labels saved successfully.'
    
    widget = Dash('Set Group Labels Widget',
                  external_stylesheets=[dbc.themes.BOOTSTRAP])
    widget.layout = layout
    return widget

def _label_syllables_widget(syllables: list[dict[str, Any]], save_syllable_info: Callable[list[dict[str, Any]], None]):
    """
    Creates a widget for labeling each syllable with a qualitative label based on it's grid movie.

    Parameters
    ----------
    syllables: list[dict[str, Any]]
        A list of dictionaries, each containing information about one syllable. Dictionary keys are:
        "syllable_id": int - The syllable's global ID. Required.
        "syllable": int - The integer label of the sylalble. Required.
        "grid_movie_path": PathLike - The path to the grid movie for this syllable. Can be an empty string.
        "label": str - The qualitative label for this syllable. Can be empty, and then this syllable will be excluded
            from the labeling process.
        "short_description": str - A longer description and notes on the syllable. Can be empty.

    save_syllable_info: Callable[list[dict[str, Any]], None]
        A list of dictionaries, each containing informatino about one syllable. Dictionary keys are:
        "syllable_id": int - The syllable's global ID. Required.
        "label": str - The qualitative label for this syllable. Can be empty, and then this syllable will be excluded
            from the labeling process.
        "short_description": str - A longer description and notes on the syllable. Can be empty.
    """
    logging.debug(f'Starting syllable labels widget with {len(syllables)} syllables')
    
    # Filter syllables that have video paths for dropdown
    syllables_with_videos = [
        s for s in syllables 
        if s.get('grid_movie_path', '') and str(s.get('grid_movie_path', '')).strip()
    ]
    
    dropdown_options = [
        {'label': f'Syllable {s["syllable"]}', 'value': s['syllable']} 
        for s in syllables_with_videos
    ]
    
    syllable_dropdown = dcc.Dropdown(
        id='syllable-dropdown',
        options=dropdown_options,
        value=dropdown_options[0]['value'] if dropdown_options else None,
        placeholder='Select a syllable to view video'
    )
    
    save_syllable_info_button = dbc.Button('Save Syllable Info', id='save-syllable-info-button')
    
    default_user_message = "Click to save syllable info"
    user_message = html.Div(id='user-message', children=default_user_message)
    
    video_player = dp.DashPlayer(
        id='video-player',
        url='',
        controls=True,
        playing=False,
        width='800px',  # Will be updated dynamically based on video
        height='600px'
    )
    
    # Prepare table data - only include syllables with grid movies
    table_records = [
        {
            'syllable_id': s['syllable_id'],  # Hidden from user but included in data
            'syllable': s['syllable'],
            'label': s.get('label', ''),
            'short_description': s.get('short_description', '')
        }
        for s in syllables_with_videos
    ]
    
    columns = [
        {'name': 'Syllable', 'id': 'syllable', 'editable': False},
        {'name': 'Label', 'id': 'label', 'editable': True, 'clearable': True},
        {'name': 'Short Description', 'id': 'short_description', 'editable': True, 'clearable': True}
    ]
    
    syllable_info_table = DataTable(
        id='syllable-labels-table',
        data=table_records,
        columns=columns,
        editable=True,
        style_cell_conditional=[
            {'if': {'column_id': 'syllable'}, 'width': '15%'},
            {'if': {'column_id': 'syllable'}, 'textAlign': 'left'},
            {'if': {'column_id': 'label'}, 'width': '25%'},
            {'if': {'column_id': 'short_description'}, 'width': '60%'}
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': '1px solid #ddd',
            'padding': '12px'
        },
        style_cell={
            'padding': '10px',
            'border': '1px solid #ddd',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_table={
            'overflowX': 'auto',
            'boxShadow': '0 0 20px rgba(0, 0, 0, 0.1)',
            'borderRadius': '8px'
        }
    )

    @callback(
        Output('user-message', 'children'),
        Input('save-syllable-info-button', 'n_clicks'),
        State('syllable-labels-table', 'data'),
        prevent_initial_call=True
    )
    def _save_syllable_info(_, table_data):
        syllable_info = []
        for row in table_data:
            syllable_info.append({
                'syllable_id': row['syllable_id'],
                'label': row['label'],
                'short_description': row['short_description']
                })

        logging.debug(f'Saving syllable info {syllable_info}')
        save_syllable_info(syllable_info)
        return 'Syllable Info Saved Successfully'

    @callback(
        [Output('video-player', 'url'),
         Output('video-player', 'width'),
         Output('video-player', 'height')],
        Input('syllable-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _update_video_player(selected_syllable):
        if selected_syllable is None:
            return '', '800px', '600px'
        
        # Find the syllable with matching ID and return its video path
        for syllable in syllables_with_videos:
            if syllable['syllable'] == selected_syllable:
                video_path = syllable['grid_movie_path']
                
                # Check if file exists
                if os.path.exists(video_path):
                    # Get video dimensions
                    width, height = get_video_dimensions(video_path)
                    
                    # Convert absolute path to URL using query parameter approach
                    import urllib.parse
                    encoded_path = urllib.parse.quote(video_path, safe='')
                    video_url = f"/video?path={encoded_path}"
                    
                    return video_url, f"{width}px", f"{height}px"
                
                return '', '800px', '600px'
        
        return '', '800px', '600px'

    layout = dbc.Row([
        dbc.Col(
            [dbc.Row([dbc.Col(syllable_dropdown), dbc.Col(save_syllable_info_button)]), dbc.Row(user_message), dbc.Row(video_player)]
        ),
        dbc.Col(syllable_info_table)
    ], style={'margin': '20px'})

    widget = Dash('Set Syllable Labels Widget',
                external_stylesheets=[dbc.themes.BOOTSTRAP])
    widget.layout = layout
    
    # Add route to serve video files using query parameters
    @widget.server.route('/video', methods=['GET'])
    def serve_video():
        """Serve video files from the filesystem using query parameter"""
        # Get filepath from query parameter
        filepath = flask.request.args.get('path')
        if not filepath:
            return "No path parameter", 400
        
        # Decode the filepath (it comes URL-encoded)
        import urllib.parse
        decoded_path = urllib.parse.unquote(filepath)
        
        if os.path.exists(decoded_path) and decoded_path.endswith('.mp4'):
            try:
                from flask import Response
                
                def generate():
                    with open(decoded_path, 'rb') as f:
                        data = f.read(1024)
                        while data:
                            yield data
                            data = f.read(1024)
                
                response = Response(generate(), mimetype='video/mp4')
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Content-Length'] = str(os.path.getsize(decoded_path))
                response.headers['Content-Type'] = 'video/mp4'
                return response
            except Exception as e:
                return f"Error serving video: {e}", 500
        else:
            return "Video not found", 404
    
    return widget

class JupyterDisplay:
    def start_set_group_labels(self, initial_group_labels: Mapping[str, str], save_group_labels: Callable[[dict[str, str]], None]):
        """Runs the widget for labeling each recording session with an experimental group label.

        Parameters
        ----------
        initial_group_labels: Mapping[str, str]
            A mapping from recording session names to experimental group labels

        save_group_labels: Callable[[dict[str, str]], None]
            Callback function that runs when 'Save Group Assignments' is clicked.
            The input parameter has the same structure as initial_group_labels:
            keys are recording session names and values are experimental group names.
        """
        port = 8050
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)

        widget = _set_group_labels_widget(initial_group_labels, save_group_labels)
        widget.run(
            host=ip,
            port=port,
            debug=False,
            use_reloader=False,
            dev_tools_ui=False,
            jupyter_mode='external'
        )

    def start_label_syllables(self, syllables: list[dict[str, Any]], save_syllable_info: Callable[list[dict[str, Any]], None]):
        """Runs the widget for labeling each syllable with a qualitative label based on its grid movie.

        Parameters
        ----------
        syllables: list[dict[str, Any]]
            A list of dictionaries, each containing information about one syllable. Dictionary keys are:
            "syllable": int - The integer label of the syllable. Required.
            "grid_movie_path": PathLike - The path to the grid movie for this syllable. Can be an empty string.
            "label": str - The qualitative label for this syllable. Can be empty.
            "short_description": str - A longer description and notes on the syllable. Can be empty.

        save_syllable_info: Callable[list[dict[str, Any]], None]
            A function that takes in the same data structure as the 'syllables' parameter, minus the 'grid_movie_path' key,
            and saves the data to disk.
        """
        port = 8051
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)

        widget = _label_syllables_widget(syllables, save_syllable_info)
        widget.run(
            host=ip,
            port=port,
            debug=False,
            use_reloader=False,
            dev_tools_ui=False,
            jupyter_mode='external'
        )

    def display_group_syllable_differences_plot(
        self,
        centers: np.ndarray,
        errors: np.ndarray,
        significant: list[list[tuple[int, int]]],
        group_labels: list[str],
        syllables: list[str | int],
        y_axis_label: str
    ):
        fig = _group_syllable_differences_plot(
            centers, errors, significant, group_labels, syllables, y_axis_label
        )
        display(fig)

# Manual visual inspection test cases - run individually as needed

def test_simple_2group():
    """Test case 1: Simple 2-group comparison with 3 syllables"""
    import numpy as np
    
    # Test data: 3 syllables, 2 groups
    centers = np.array([
        [0.8, 1.2],  # syllable 0
        [1.5, 0.9],  # syllable 1  
        [0.6, 0.7]   # syllable 2
    ])
    errors = np.array([
        [0.1, 0.15],
        [0.2, 0.1], 
        [0.08, 0.12]
    ])
    significant = [
        [(0, 1)],  # syllable 0: groups 0 vs 1 significant
        [],        # syllable 1: no significance
        []         # syllable 2: no significance  
    ]
    group_labels = ['Control', 'Treatment']
    syllables = ['0', '1', '2']
    y_axis_label = 'Mean Usage Rate'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_simple_2group.png', dpi=150, bbox_inches='tight')
    print("Saved: test_simple_2group.png")

def test_multi_group():
    """Test case 2: Multi-group comparison with mixed significance patterns"""
    import numpy as np
    
    # Test data: 4 syllables, 4 groups
    centers = np.array([
        [1.2, 0.8, 1.5, 0.9],  # syllable 0
        [0.7, 1.1, 0.6, 1.3],  # syllable 1
        [1.0, 1.4, 0.8, 1.1],  # syllable 2
        [0.9, 0.5, 1.2, 0.8]   # syllable 3
    ])
    errors = np.array([
        [0.12, 0.08, 0.18, 0.10],
        [0.09, 0.14, 0.07, 0.16],
        [0.11, 0.20, 0.09, 0.13],
        [0.10, 0.06, 0.15, 0.09]
    ])
    significant = [
        [(0, 2), (1, 3)],  # syllable 0: two comparisons
        [(0, 1)],          # syllable 1: one comparison
        [],                # syllable 2: no significance
        [(1, 2), (2, 3), (0, 3)]  # syllable 3: three comparisons (test stacking)
    ]
    group_labels = ['Wild-type', 'Mutant A', 'Mutant B', 'Rescue']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Frequency (Hz)'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_multi_group.png', dpi=150, bbox_inches='tight')
    print("Saved: test_multi_group.png")

def test_many_syllables():
    """Test case 3: Many syllables with sparse significance"""
    import numpy as np
    
    # Test data: 57 syllables, 3 groups
    np.random.seed(42)  # For reproducible test data
    centers = np.random.randn(57, 3) * 0.3 + 1.0  # centered around 1.0
    errors = np.abs(np.random.randn(57, 3)) * 0.1 + 0.05  # small positive errors
    
    # Sparse significance - about 10% of syllables significant
    significant = [[] for _ in range(57)]  # Initialize all empty
    significant[3] = [(0, 1)]
    significant[12] = [(1, 2)]
    significant[23] = [(0, 2)]
    significant[34] = [(0, 1)]
    significant[45] = [(1, 2)]
    significant[52] = [(0, 2)]
    
    group_labels = ['Group A', 'Group B', 'Group C']
    syllables = [str(i) for i in range(57)]
    y_axis_label = 'Normalized Expression'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_many_syllables.png', dpi=150, bbox_inches='tight')
    print("Saved: test_many_syllables.png")

def test_single_syllable():
    """Test case 4: Single syllable with multiple groups and stacked significance"""
    import numpy as np
    
    # Test data: 1 syllable, 4 groups
    centers = np.array([[1.2, 0.8, 1.5, 0.6]])  # shape: (1, 4)
    errors = np.array([[0.15, 0.12, 0.18, 0.10]])
    
    # Multiple significant comparisons to test vertical stacking
    significant = [
        [(0, 1), (0, 2), (1, 3), (2, 3)]  # 4 comparisons for stacking test
    ]
    group_labels = ['Baseline', 'Low Dose', 'High Dose', 'Recovery']
    syllables = ['0']
    y_axis_label = 'Response Magnitude'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_single_syllable.png', dpi=150, bbox_inches='tight')
    print("Saved: test_single_syllable.png")

def test_no_significance():
    """Test case 5: No significance - all empty lists"""
    import numpy as np
    
    # Test data: 4 syllables, 3 groups, no significance
    centers = np.array([
        [1.1, 0.9, 1.3],
        [0.8, 1.2, 0.7],
        [1.4, 1.0, 1.1],
        [0.6, 0.9, 1.2]
    ])
    errors = np.array([
        [0.12, 0.10, 0.15],
        [0.08, 0.14, 0.09],
        [0.16, 0.11, 0.13],
        [0.07, 0.10, 0.14]
    ])
    significant = [[], [], [], []]  # No significance anywhere
    group_labels = ['Group X', 'Group Y', 'Group Z']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Activity Level'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_no_significance.png', dpi=150, bbox_inches='tight')
    print("Saved: test_no_significance.png")

def test_all_significant():
    """Test case 6: Every possible group comparison significant for every syllable"""
    import numpy as np
    
    # Test data: 3 syllables, 3 groups (3 possible pairwise comparisons per syllable)
    centers = np.array([
        [1.5, 0.8, 1.2],
        [0.9, 1.4, 0.7],
        [1.1, 0.6, 1.3]
    ])
    errors = np.array([
        [0.15, 0.08, 0.12],
        [0.09, 0.14, 0.07],
        [0.11, 0.06, 0.13]
    ])
    # All possible comparisons for 3 groups: (0,1), (0,2), (1,2)
    significant = [
        [(0, 1), (0, 2), (1, 2)],  # syllable 0: all comparisons
        [(0, 1), (0, 2), (1, 2)],  # syllable 1: all comparisons
        [(0, 1), (0, 2), (1, 2)]   # syllable 2: all comparisons
    ]
    group_labels = ['Control', 'Treatment A', 'Treatment B']
    syllables = ['0', '1', '2']
    y_axis_label = 'Response Score'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_all_significant.png', dpi=150, bbox_inches='tight')
    print("Saved: test_all_significant.png")

def test_large_error_bars():
    """Test case 7: Large error bars (50%+ of center values)"""
    import numpy as np
    
    # Test data: 3 syllables, 3 groups with very large error bars
    centers = np.array([
        [2.0, 1.5, 2.2],
        [1.8, 2.1, 1.4],
        [1.6, 1.9, 2.0]
    ])
    # Error bars that are 50-80% of center values
    errors = np.array([
        [1.0, 0.9, 1.3],  # 50%, 60%, 59% of centers
        [1.1, 1.2, 0.8],  # 61%, 57%, 57% of centers  
        [0.9, 1.4, 1.1]   # 56%, 74%, 55% of centers
    ])
    significant = [
        [(0, 1)],  # syllable 0
        [],        # syllable 1
        [(1, 2)]   # syllable 2
    ]
    group_labels = ['Condition 1', 'Condition 2', 'Condition 3']
    syllables = ['0', '1', '2']
    y_axis_label = 'Variable Measure'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_large_error_bars.png', dpi=150, bbox_inches='tight')
    print("Saved: test_large_error_bars.png")

def test_many_stacked_bars():
    """Test case 8: One syllable with many stacked significance bars"""
    import numpy as np
    
    # Test data: 2 syllables, 6 groups to create many possible comparisons
    centers = np.array([
        [1.0, 1.3, 0.8, 1.5, 0.7, 1.2],  # syllable 0
        [0.9, 1.1, 1.0, 0.8, 1.2, 0.95]  # syllable 1
    ])
    errors = np.array([
        [0.1, 0.13, 0.08, 0.15, 0.07, 0.12],
        [0.09, 0.11, 0.10, 0.08, 0.12, 0.095]
    ])
    significant = [
        [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 5)],  # 6 comparisons - heavy stacking
        [(1, 2)]  # syllable 1: just one for contrast
    ]
    group_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    syllables = ['0', '1']
    y_axis_label = 'Signal Intensity'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_many_stacked_bars.png', dpi=150, bbox_inches='tight')
    print("Saved: test_many_stacked_bars.png")

def test_minimal_spacing():
    """Test case 9: Groups with very similar values (test visual separation)"""
    import numpy as np
    
    # Test data: 4 syllables, 4 groups with very similar values
    centers = np.array([
        [1.000, 1.002, 1.001, 1.003],  # Very close values
        [0.998, 1.001, 0.999, 1.000],
        [1.001, 0.999, 1.002, 0.998],
        [1.002, 1.000, 0.999, 1.001]
    ])
    errors = np.array([
        [0.0005, 0.0008, 0.0006, 0.0007],  # Small errors relative to differences
        [0.0006, 0.0009, 0.0005, 0.0008],
        [0.0007, 0.0005, 0.0008, 0.0006],
        [0.0008, 0.0006, 0.0005, 0.0009]
    ])
    significant = [
        [(0, 3)],  # syllable 0
        [],        # syllable 1
        [(1, 3)],  # syllable 2
        [(0, 2)]   # syllable 3
    ]
    group_labels = ['Type I', 'Type II', 'Type III', 'Type IV']
    syllables = ['0', '1', '2', '3']
    y_axis_label = 'Precise Measurement'
    
    fig = _group_syllable_differences_plot(
        centers, errors, significant, group_labels, syllables, y_axis_label
    )
    fig.savefig('test_minimal_spacing.png', dpi=150, bbox_inches='tight')
    print("Saved: test_minimal_spacing.png")

# Uncomment and run any test case:
test_simple_2group()
test_multi_group() 
test_many_syllables()
test_single_syllable()
test_no_significance()
test_all_significant()
test_large_error_bars()
test_many_stacked_bars()
test_minimal_spacing()