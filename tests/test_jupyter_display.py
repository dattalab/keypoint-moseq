import pytest
import time
import keypoint_moseq as kpms
import polars as pl
from unittest.mock import Mock
from keypoint_moseq.view.jupyter_display import _set_group_labels_widget
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

@pytest.fixture
def initial_labels():
    return {'mouse0': '', 'mouse1': '', 'mouse2': ''}

def edit_and_assert_cell(dash_duo, row_idx, value, expected_values):
    """
    Edit a cell in the Dash DataTable and assert the expected values.

    Data table assumed to have the id group-labels-table

    Parameters
    ----------
    dash_duo : DashDuo
        The DashDuo test instance
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
    cell_input_text_length = len(cell_input.get_attribute('value'))
    for _ in range(cell_input_text_length):
        cell_input.send_keys(Keys.BACKSPACE)

    cell_input.send_keys(value)
    first_col.click()

    cell_value_divs = dash_duo.find_elements('#group-labels-table tbody tr td div.dash-cell-value')
    cell_text_values = [div.text for div in cell_value_divs][1::2]
    assert cell_text_values == expected_values

def test_set_group_labels_widget_mock(initial_labels, dash_duo):
    save_group_labels_mock = Mock()
    widget = _set_group_labels_widget(initial_labels, save_group_labels_mock)

    dash_duo.start_server(widget)
    dash_duo.wait_for_text_to_equal('#user-message', 'Click to save group assignments')

    # Test that the table initialized properly
    rows = dash_duo.find_elements('#group-labels-table tbody tr')
    assert [r.text for r in rows[1:]] == ['mouse0', 'mouse1', 'mouse2']
    save_group_labels_mock.assert_not_called()

    # Test editing the table
    edit_and_assert_cell(dash_duo, 1, 'WT', ['WT', '', ''])
    edit_and_assert_cell(dash_duo, 2, 'HET', ['WT', 'HET', ''])
    edit_and_assert_cell(dash_duo, 3, 'WT', ['WT', 'HET', 'WT'])
    edit_and_assert_cell(dash_duo, 1, 'HET', ['HET', 'HET', 'WT'])

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

def test_set_group_labels_tutorial_data_e2e(demo_project_dir, deeplabcut_2d_zenodo_dir, dash_duo):
    dash_duo.driver.set_window_size(1920, 1080)

    config = lambda: kpms.load_config(demo_project_dir)
    dlc_config = str(deeplabcut_2d_zenodo_dir / 'dlc_project/config.yaml')
    video_dir = str(deeplabcut_2d_zenodo_dir / 'dlc_project/videos/')
    recordings_csv_path = demo_project_dir / 'recordings.csv'

    project = kpms.KPMSProject(demo_project_dir)
    disp = kpms.JupyterDisplay()
    c = kpms.Controller(project, disp)

    kpms.setup_project(demo_project_dir, deeplabcut_config=dlc_config)
    kpms.update_config(
        demo_project_dir,
        video_dir=video_dir,
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4'],
        use_bodyparts=['spine4', 'spine3', 'spine2', 'spine1', 'head',
                    'nose', 'right ear', 'left ear'],
        fps=30
    )

    _ = c.load_keypoints(video_dir, 'deeplabcut')

    # The new load_keypoints should create this file
    assert recordings_csv_path.exists()
    recordings_df = pl.read_csv(recordings_csv_path)

    # There should only be the 'name' column after creation
    assert 'name' in recordings_df.columns
    assert len(recordings_df.columns) == 1

    initial_labels = {name: '' for name in recordings_df.get_column('name')}

    widget = _set_group_labels_widget(initial_labels, c._save_group_labels)
    dash_duo.start_server(widget)
    dash_duo.wait_for_text_to_equal('#user-message', 'Click to save group assignments')

    num_recordings = len(recordings_df)

    edit_and_assert_cell(dash_duo, 1, 'WT', ['WT'] + ([''] * (num_recordings-1)))
    edit_and_assert_cell(dash_duo, 2, 'HET', ['WT', 'HET'] + ([''] * (num_recordings-2)))
    edit_and_assert_cell(dash_duo, 3, 'WT', ['WT', 'HET', 'WT'] + ([''] * (num_recordings-3)))
    edit_and_assert_cell(dash_duo, 1, 'HET', ['HET', 'HET', 'WT'] + ([''] * (num_recordings-3)))

    save_button = dash_duo.find_element('#save-button')
    save_button.click()
    time.sleep(2)

    recordings_df = pl.read_csv(recordings_csv_path)

    # 'group' column should have been added on save
    assert 'group' in recordings_df.columns
    assert len(recordings_df.columns) == 2

    assert recordings_df.get_column('group').to_list() == ['HET', 'HET', 'WT'] + ([''] * (num_recordings-3))

    edit_and_assert_cell(dash_duo, 3, '', ['HET', 'HET'] + ([''] * (num_recordings-2)))

    save_button.click()
    time.sleep(2)

    recordings_df = pl.read_csv(recordings_csv_path)
    assert recordings_df.get_column('group').to_list() == ['HET', 'HET'] + ([''] * (num_recordings-2))