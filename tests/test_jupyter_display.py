import pytest
import time
from unittest.mock import Mock
from keypoint_moseq.view.jupyter_display import _set_group_labels_widget
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

@pytest.fixture
def initial_labels():
    return {'mouse0': '', 'mouse1': '', 'mouse2': ''}

def test_set_group_labels_widget_mock(initial_labels, dash_duo):
    def edit_and_assert_cell(row_idx, value, expected_values):
        """
        Edit a cell in the Dash DataTable and assert the expected values.

        Data table assumed to have the id group-labels-table

        Parameters
        ----------
        row_idx : int
            1-based index of the row to edit (excluding header).
        value : str
            The value to enter into the cell.
        expected_values : list of str
            The expected values for the group-label column after editing.
        """
        first_col = dash_duo.find_element(f'#group-labels-table tbody tr:nth-child({row_idx+1}) td:nth-child(1)')
        cell = dash_duo.find_element(f'#group-labels-table tbody tr:nth-child({row_idx+1}) td:nth-child(2)')
        cell.click()
        cell_input = cell.find_element_by_css_selector('input.dash-cell-value')
        cell_input.clear()
        cell_input.send_keys(value)
        first_col.click()
        cell_value_divs = dash_duo.find_elements('#group-labels-table tbody tr td div.dash-cell-value')
        cell_text_values = [div.text for div in cell_value_divs][1::2]
        assert cell_text_values == expected_values

    save_group_labels_mock = Mock()
    widget = _set_group_labels_widget(initial_labels, save_group_labels_mock)

    dash_duo.start_server(widget)
    dash_duo.wait_for_text_to_equal('#user-message', 'Click to save group assignments')

    # Test that the table initialized properly
    rows = dash_duo.find_elements('#group-labels-table tbody tr')
    assert [r.text for r in rows[1:]] == ['mouse0', 'mouse1', 'mouse2']
    save_group_labels_mock.assert_not_called()

    # Test editing the table
    edit_and_assert_cell(1, 'WT', ['WT', '', ''])
    edit_and_assert_cell(2, 'HET', ['WT', 'HET', ''])
    edit_and_assert_cell(3, 'WT', ['WT', 'HET', 'WT'])
    edit_and_assert_cell(1, 'HET', ['HET', 'HET', 'WT'])

    # Save callback should not have been called up to this point
    save_group_labels_mock.assert_not_called()

    save_button = dash_duo.find_element('#save-button')
    save_button.click()
    time.sleep(0.5)

    # Now the save callback should have been called with the current state of the table
    save_group_labels_mock.assert_called_once_with({
        'mouse0': 'HET',
        'mouse1': 'HET',
        'mouse2': 'WT',
    })

def test_save_group_labels_callback(initial_labels, dash_duo):
    save_group_labels_mock = Mock()
    widget = _set_group_labels_widget(initial_labels, save_group_labels_mock)
    dash_duo.start_server(widget)
    dash_duo.wait_for_text_to_equal('#user-message', 'Click to save group assignments')

    callback = widget.callback_map['user-message.children']['callback'].__wrapped__

    n_clicks = 1
    bad_table_records = [
        {'recording-name': 'mouse0', 'group-label': 'WT'},
        {'recording-name': 'mouse1', 'group-label': 'HET'},
        {'recording-name': 'mouse2', 'group-label': 'WT'},
        {'recording-name': '', 'group-label': 'WT'},
        {'recording-name': '', 'group-label': 'HET'}
    ]

    result = callback(n_clicks, bad_table_records)
    assert result == 'Labels saved successfully.'

    save_group_labels_mock.assert_called_once_with({
        'mouse0': 'WT',
        'mouse1': 'HET',
        'mouse2': 'WT'
    })

    result = callback(n_clicks, [])
    assert result == 'Labels saved successfully.'

    save_group_labels_mock.assert_called_with({})

    
    