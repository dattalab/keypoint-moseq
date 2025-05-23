from dash.dash_table import DataTable
from dash import html, callback, Input, Output, State
from typing import Callable, Mapping
from IPython.display import display

class JupyterDisplay:
    def start_set_group_labels(self, initial_group_labels: Mapping[str, str], save_group_labels: Callable[[dict[str, str]]]):
        """Runs the widget for labeling each recording session with an experimental group label.

        Parameters
        ----------
        initial_group_labels: Mapping[str, str]
            A mapping from recording session names to experimental group labels

        save_group_labels: Callable[dict[str, str]]
            Callback function that runs when 'Save Group Assignments' is clicked.
            The input parameter has the same structure as initial_group_labels:
            keys are recording session names and values are experimental group names.
        """
        records = [
            {'Recording Name': name, 'Group Label': group} 
            for name, group in initial_group_labels.items()
        ]

        columns = [
            {'name': 'Recording Name', 'id': 'recording-name', 'clearable': False, 'editable': False},
            {'name': 'Group Label', 'id': 'group-label', 'clearable': True, 'editable': True}
        ]

        user_message = html.Div(id='user-message', children="")
        table = DataTable(id='group-labels-table', data=records, columns=columns)
        save_button = html.Button('Save Group Assignments', id='save-button')

        @callback(
            Output('user-message', 'children'),
            Input('save-button', 'n_clicks'),
            State('group-labels-table', 'data')
        )
        def _save_group_labels(n_clicks, table_data):
            if not n_clicks:
                return

            group_labels = {
                row['Recording Name']: row['Group Label']
                for row in table_data
            }

            save_group_labels(group_labels)
            return "Labels saved succesfully."
        
        display(html.Div([user_message, table, save_button]))





