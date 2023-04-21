"""
This module contains the widget components that comprise the group setting table functionality.
"""

import qgrid
import pandas as pd
import yaml
import ipywidgets as widgets
from IPython.display import clear_output
from keypoint_moseq.analysis import index_to_dataframe

class GroupSettingWidgets:

    def __init__(self, index_filepath):
        """
        Initialize all the Group Setting widgets, parses the index yaml file into a pandas DataFrame that is compatible to be displayed using QGrid.

        Args:
        index_filepath (str): Path to index file (moseq2-index.yaml) containing session metadata and grouping info.
        """

        self.index_filepath = index_filepath
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

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

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', style=style,
                                        description='New Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group Name', style=style,
                                          disabled=False, tooltip='Set Group')
        self.update_index_button = widgets.Button(description='Update Index File', style=style,
                                                  disabled=False, tooltip='Save Parameters')

        self.group_set = widgets.HBox([self.group_input, self.save_button, self.update_index_button])

        self.index_dict, self.df = index_to_dataframe(self.index_filepath)
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

        updated_index = {'files': list(self.df.to_dict(orient='index').values())}

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