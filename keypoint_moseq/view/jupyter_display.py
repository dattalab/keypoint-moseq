import dash_bootstrap_components as dbc
import logging
from dash.dash_table import DataTable
from dash import Dash, html, callback, Input, Output, State
from typing import Callable, Mapping

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

    default_user_message = "Click to save group assignments"

    user_message = html.Div(id='user-message', children=default_user_message)
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

    layout = dbc.Row(
        [dbc.Col(table), dbc.Col([user_message, save_button])],
        style={'margin': '20px'})

    @callback(
        Output('user-message', 'children'),
        Input('save-button', 'n_clicks'),
        State('group-labels-table', 'data')
    )
    def _save_group_labels(n_clicks, table_data):
        if not n_clicks:
            return default_user_message

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
        widget = _set_group_labels_widget(initial_group_labels, save_group_labels)
        widget.run(jupyter_mode='inline')