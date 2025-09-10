import pytest
import keypoint_moseq as kpms
import polars as pl
from pathlib import Path
from unittest.mock import Mock
from keypoint_moseq.project.kpms_project import KPMSProject
from keypoint_moseq.view.jupyter_display import JupyterDisplay
from keypoint_moseq.controller.controller import Controller

@pytest.fixture
def project():
    mock_project = Mock(spec=KPMSProject)
    mock_project.log_dir_path = Path('/tmp/logs')
    return mock_project

@pytest.fixture 
def display():
    return Mock(spec=JupyterDisplay)

def test_init(project, display):
    c = Controller(project, display)
    assert c.project == project
    assert c.display == display

def test_save_group_labels_typical_data(project, display):
    c = Controller(project, display)
    group_labels = {
        'recording1': 'control',
        'recording2': 'treatment',
        'recording3': 'control'
    }
    c._save_group_labels(group_labels)
    project.update_recordings.assert_called_once()
    recordings_updates_df: pl.DataFrame = project.update_recordings.call_args[0][0]
    assert 'name' in recordings_updates_df.columns
    assert 'group' in recordings_updates_df.columns
    assert len(recordings_updates_df) == 3
    for name, group in group_labels.items():
        row = recordings_updates_df.filter(pl.col('name') == name)
        assert len(row) == 1
        assert row.get_column('group').item() == group

def test_save_group_labels_empty_data(project, display):
    c = Controller(project, display)
    group_labels = {}
    c._save_group_labels(group_labels)
    project.update_recordings.assert_called_once()
    call_args = project.update_recordings.call_args[0][0]
    assert 'name' in call_args.columns
    assert 'group' in call_args.columns
    assert len(call_args) == 0

def test_set_group_labels_with_existing_groups(project, display):
    recordings = pl.DataFrame({
        'name': ['rec1', 'rec2', 'rec3'],
        'group': ['control', 'treatment', 'control']
    })
    project.get_recordings.return_value = recordings
    c = Controller(project, display)
    c.set_group_labels()
    display.start_set_group_labels.assert_called_once()
    initial_labels, callback = display.start_set_group_labels.call_args[0]
    assert initial_labels == {'rec1': 'control', 'rec2': 'treatment', 'rec3': 'control'}
    assert callback == c._save_group_labels

def test_set_group_labels_without_groups(project, display):
    recordings = pl.DataFrame({
        'name': ['rec1', 'rec2', 'rec3']
    })
    project.get_recordings.return_value = recordings
    c = Controller(project, display)
    c.set_group_labels()
    display.start_set_group_labels.assert_called_once()
    initial_labels, callback = display.start_set_group_labels.call_args[0]
    assert initial_labels == {'rec1': '', 'rec2': '', 'rec3': ''}
    assert callback == c._save_group_labels

def test_set_group_labels_empty_recordings(project, display):
    recordings = pl.DataFrame({
        'name': []
    })
    project.get_recordings.return_value = recordings
    c = Controller(project, display)
    c.set_group_labels()
    display.start_set_group_labels.assert_called_once()
    initial_labels, callback = display.start_set_group_labels.call_args[0]
    assert initial_labels == {}
    assert callback == c._save_group_labels

def test_set_group_labels_sorted_order(project, display):
    recordings = pl.DataFrame({
        'name': ['rec1', 'rec2', 'rec3'],
        'group': ['treatment', 'control', 'treatment']
    })
    project.get_recordings.return_value = recordings
    c = Controller(project, display)
    c.set_group_labels()
    display.start_set_group_labels.assert_called_once()
    initial_labels, callback = display.start_set_group_labels.call_args[0]
    assert list(initial_labels.items()) == [
        ('rec2', 'control'),
        ('rec1', 'treatment'),
        ('rec3', 'treatment')
    ]
    assert callback == c._save_group_labels

