import dash_bootstrap_components as dbc
import socket
import logging
from dash.dash_table import DataTable
from dash import Dash, html, dcc, callback, Input, Output, State
from typing import Callable, Mapping, Any
from IPython.display import IFrame, display
import ipywidgets as widgets
import dash_player as dp
import flask
import os

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