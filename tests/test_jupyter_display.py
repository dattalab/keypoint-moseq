import polars as pl
import pytest
import keypoint_moseq as kpms
import importlib
import keypoint_moseq.view.jupyter_display as jd
from unittest.mock import Mock, patch
from keypoint_moseq.view.jupyter_display import JupyterDisplay

@pytest.fixture
def initial_labels():
    return {'mouse0': '', 'mouse1': '', 'mouse2': ''}

@pytest.fixture
def mock_save_callback():
    return Mock()

@pytest.fixture
def jupyter_display():
    return JupyterDisplay()

@pytest.fixture
def has_recordings_demo_project(deeplabcut_2d_zenodo_dir, demo_project_dir):
    """
    Fixture that sets up a demo project with loaded recordings.
    Returns a tuple of (controller, project_dir) for use in tests.
    """
    config = lambda: kpms.load_config(demo_project_dir)
    project = kpms.KPMSProject(demo_project_dir)
    disp = kpms.JupyterDisplay()
    c = kpms.Controller(project, disp)
    
    dlc_config = str(deeplabcut_2d_zenodo_dir / 'dlc_project/config.yaml')
    video_dir = str(deeplabcut_2d_zenodo_dir / 'dlc_project/videos')
    kpms.setup_project(str(demo_project_dir), deeplabcut_config=dlc_config)

    kpms.update_config(
        str(demo_project_dir),
        video_dir=video_dir,
        anterior_bodyparts=['nose'],
        posterior_bodyparts=['spine4'],
        use_bodyparts=['spine4', 'spine3', 'spine2', 'spine1', 'head', 'nose', 'right ear', 'left ear'],
        fps=30
    )

    keypoint_data_path = video_dir
    _ = c.load_keypoints(keypoint_data_path, 'deeplabcut')
    
    return c, demo_project_dir

def test_save_group_labels_initial_state(jupyter_display, initial_labels, mock_save_callback):
    captured = {}

    def fake_callback(*dargs, **dkwargs):
        def decorator(func):
            captured['args'] = dargs
            captured['kwargs'] = dkwargs
            captured['func'] = func
            return func

        return decorator

    with patch('keypoint_moseq.view.jupyter_display.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('keypoint_moseq.view.jupyter_display.DataTable') as mock_table, \
         patch('keypoint_moseq.view.jupyter_display.callback', new=fake_callback) as mock_callback:

        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance

        jupyter_display.start_set_group_labels(initial_labels, mock_save_callback)

        mock_dash.assert_called_once()
        mock_table.assert_called_once()

        print(captured)

        table_args = mock_table.call_args[1]
        assert table_args['data'] == [
            {'recording-name': 'mouse0', 'group-label': ''},
            {'recording-name': 'mouse1', 'group-label': ''},
            {'recording-name': 'mouse2', 'group-label': ''}
        ]

        callback_func = captured['func']

        result = callback_func(None, [])
        assert result == "Click to save group assignments"
        mock_save_callback.assert_not_called()

def test_save_group_labels_valid_data(jupyter_display, initial_labels, mock_save_callback):
    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance
        
        jupyter_display.start_set_group_labels(initial_labels, mock_save_callback)
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        # Test saving with valid data
        table_data = [
            {'recording-name': 'mouse0', 'group-label': 'group1'},
            {'recording-name': 'mouse1', 'group-label': 'group2'},
            {'recording-name': 'mouse2', 'group-label': 'group3'}
        ]
        
        result = callback_func(1, table_data)
        assert result == "Labels saved succesfully."
        mock_save_callback.assert_called_once_with({
            'mouse0': 'group1',
            'mouse1': 'group2',
            'mouse2': 'group3'
        })

def test_save_group_labels_empty_table(jupyter_display, mock_save_callback):
    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance
        
        jupyter_display.start_set_group_labels({}, mock_save_callback)
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        result = callback_func(1, [])
        assert result == "Labels saved succesfully."
        mock_save_callback.assert_called_with({})

def test_save_group_labels_missing_group_label(jupyter_display, initial_labels, mock_save_callback):
    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance
        
        jupyter_display.start_set_group_labels(initial_labels, mock_save_callback)
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        table_data = [
            {'recording-name': 'mouse0'},
            {'recording-name': 'mouse1'},
            {'recording-name': 'mouse2'}
        ]
        
        result = callback_func(1, table_data)
        assert result == "Labels saved succesfully."
        mock_save_callback.assert_called_with({
            'mouse0': '',
            'mouse1': '',
            'mouse2': ''
        })

@pytest.mark.parametrize("initial_data,expected_table_data", [
    (
        {'mouse0': 'group1'}, 
        [{'recording-name': 'mouse0', 'group-label': 'group1'}]
    ),
    (
        {}, 
        []
    ),
    (
        {
            'mouse0': 'group1',
            'mouse1': 'group2'
        }, 
        [
            {'recording-name': 'mouse0', 'group-label': 'group1'},
            {'recording-name': 'mouse1', 'group-label': 'group2'}
        ]
    )
])
def test_set_group_labels_widget_initialization(jupyter_display, initial_data, expected_table_data):
    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance
        
        jupyter_display.start_set_group_labels(initial_data, lambda x: None)
        
        mock_table.assert_called_once()
        table_kwargs = mock_table.call_args[1]
        assert table_kwargs['data'] == expected_table_data

def test_set_group_labels_integration_no_changes(has_recordings_demo_project):
    """
    Tests that the recordings DataFrame remains unchanged when no group labels are modified.
    """
    c, demo_project_dir = has_recordings_demo_project
    print('got to test code')
    
    initial_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')

    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance

        print('before set group labels')
        c.set_group_labels()

    print('out of context manager')
        
    updated_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')

    assert initial_recordings_df.equals(updated_recordings_df), "Recordings DataFrame should be unchanged when no group labels are modified"

def test_set_group_labels_integration_set_groups(has_recordings_demo_project):
    """
    Tests that group label changes persist when saved through the callback.
    """
    c, demo_project_dir = has_recordings_demo_project
    
    initial_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')
    recording_names = initial_recordings_df.get_column('name').to_list()

    # Alternate between WT and HET groups
    new_groups = ['WT' if i % 2 == 0 else 'HET' for i in range(len(recording_names))]
    group_labels = dict(zip(recording_names, new_groups))

    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance

        c.set_group_labels()
        
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        table_data = [
            {'recording-name': name, 'group-label': group}
            for name, group in group_labels.items()
        ]
        
        callback_func(1, table_data)
        
    updated_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')

    # Check that the changes were saved
    assert not initial_recordings_df.equals(updated_recordings_df), "Recordings DataFrame should be modified after setting groups"
    assert updated_recordings_df.get_column('group').value_counts().to_dict() == {'WT': len(recording_names)//2, 'HET': (len(recording_names)+1)//2}

def test_set_group_labels_integration_overwrite_existing_groups(has_recordings_demo_project):
    """
    Tests that specific group label changes are correctly overwritten when saved through the callback.
    """
    c, demo_project_dir = has_recordings_demo_project
    
    recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')
    recording_names = recordings_df.get_column('name').to_list()
    
    initial_groups = ['Control'] * len(recording_names)
    recordings_with_groups = recordings_df.with_columns(pl.Series('group', initial_groups))
    recordings_with_groups.write_csv(demo_project_dir / 'recordings.csv')

    modified_group_labels = dict(zip(recording_names, initial_groups))
    modified_group_labels[recording_names[0]] = 'Treatment'

    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance

        c.set_group_labels()
        
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        table_data = [
            {'recording-name': name, 'group-label': group}
            for name, group in modified_group_labels.items()
        ]
        
        callback_func(1, table_data)
        
    updated_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')

    updated_groups = updated_recordings_df.get_column('group').to_list()
    assert updated_groups[0] == 'Treatment', "First recording should be changed to 'Treatment'"
    assert all(group == 'Control' for group in updated_groups[1:]), "All other recordings should remain 'Control'"
    
    group_counts = updated_recordings_df.get_column('group').value_counts().to_dict()
    assert group_counts == {'Control': len(recording_names) - 1, 'Treatment': 1}

def test_set_group_labels_integration_ignores_empty_recording_names(has_recordings_demo_project):
    """
    Tests that the callback ignores records with empty 'recording-name' values and doesn't
    add extra rows to the recordings CSV.
    """
    c, demo_project_dir = has_recordings_demo_project
    
    initial_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')
    recording_names = initial_recordings_df.get_column('name').to_list()
    initial_row_count = len(initial_recordings_df)

    with patch('dash.Dash') as mock_dash, \
         patch('dash_bootstrap_components.Button') as mock_button, \
         patch('dash.dash_table.DataTable') as mock_table:
        
        mock_dash_instance = Mock()
        mock_dash.return_value = mock_dash_instance

        c.set_group_labels()
        
        callback_func = mock_dash_instance.callback.call_args[0][0]
        
        table_data = [
            {'recording-name': name, 'group-label': 'Control'}
            for name in recording_names
        ]
        table_data.extend([
            {'recording-name': '', 'group-label': 'EmptyGroup1'},
            {'recording-name': '', 'group-label': 'EmptyGroup2'},
            {'recording-name': '', 'group-label': ''}
        ])
        
        callback_func(1, table_data)
        
    updated_recordings_df = pl.read_csv(demo_project_dir / 'recordings.csv')

    assert len(updated_recordings_df) == initial_row_count, "No extra rows should be added to recordings CSV"
    
    updated_names = updated_recordings_df.get_column('name').to_list()
    assert set(updated_names) == set(recording_names), "All original recording names should be preserved"
    assert all(group == 'Control' for group in updated_recordings_df.get_column('group').to_list()), "All valid recordings should have 'Control' group"