import numpy as np
import polars as pl
import pytest
from keypoint_moseq.project.kpms_project import _check_id_column_integrity, _update_dataframe
from keypoint_moseq.project.kpms_project import KPMSProject

def test_check_id_column_integrity_ids_ok():
    df = pl.DataFrame({'id': [1, 2, 3], 'value': ['hello', 'there', 'world']})
    _check_id_column_integrity(df, 'id')  # Should not raise

def test_check_id_column_integrity_missing_column():
    df = pl.DataFrame({'value': ['hello', 'there', 'world']})
    with pytest.raises(ValueError, match='Dataframe must have column id_column "id"'):
        _check_id_column_integrity(df, 'id')

def test_check_id_column_integrity_duplicate_ids():
    df = pl.DataFrame({'id': [1, 2, 2], 'value': ['hello', 'there', 'world']})
    with pytest.raises(ValueError, match='Dataframe contains duplicate values in id_column id: \\[2\\]'):
        _check_id_column_integrity(df, 'id')

def test_check_id_column_integrity_empty_dataframe():
    df = pl.DataFrame({'id': [], 'value': []})
    _check_id_column_integrity(df, 'id')  # Should not raise - empty dataframe has no duplicates

def test_check_id_column_integrity_single_row():
    df = pl.DataFrame({'id': [1], 'value': ['hello']})
    _check_id_column_integrity(df, 'id')  # Should not raise - single row has no duplicates

def test_check_id_column_integrity_different_column_name():
    df = pl.DataFrame({'session_id': [1, 2, 3], 'value': ['hello', 'there', 'world']})
    _check_id_column_integrity(df, 'session_id')  # Should not raise

def test_check_id_column_integrity_multiple_columns_no_id():
    df = pl.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [True, False, True]
    })
    with pytest.raises(ValueError, match='Dataframe must have column id_column "id". Got columns: \\[\'col1\', \'col2\', \'col3\'\\]'):
        _check_id_column_integrity(df, 'id')

# Fixtures for _update_dataframe tests
@pytest.fixture
def base_dataframe():
    return pl.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })

@pytest.fixture
def update_dataframe():
    return pl.DataFrame({
        'id': [1, 3],
        'name': ['Alice Updated', 'Charlie Updated'],
        'new_col': ['new1', 'new2']
    })

def test_update_dataframe_basic_update(base_dataframe, update_dataframe):
    result = _update_dataframe(base_dataframe, update_dataframe, 'id')
    
    # Check that rows 1 and 3 were updated
    assert result.filter(pl.col('id') == 1)['name'].item() == 'Alice Updated'
    assert result.filter(pl.col('id') == 3)['name'].item() == 'Charlie Updated'
    
    # Check that row 2 was preserved
    assert result.filter(pl.col('id') == 2)['name'].item() == 'Bob'
    
    # Check that new column was added with null for non-updated row
    assert result.filter(pl.col('id') == 1)['new_col'].item() == 'new1'
    assert result.filter(pl.col('id') == 2)['new_col'].is_null().item()
    assert result.filter(pl.col('id') == 3)['new_col'].item() == 'new2'

def test_update_dataframe_missing_id_column():
    df1 = pl.DataFrame({'name': ['Alice', 'Bob']})
    df2 = pl.DataFrame({'name': ['Alice Updated']})
    
    with pytest.raises(ValueError, match='Dataframe must have column id_column "id"'):
        _update_dataframe(df1, df2, 'id')

def test_update_dataframe_duplicate_ids():
    df1 = pl.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
    df2 = pl.DataFrame({'id': [1, 1], 'name': ['Alice Updated', 'Alice Updated Again']})
    
    with pytest.raises(ValueError, match='Dataframe contains duplicate values in id_column id: \\[1\\]'):
        _update_dataframe(df1, df2, 'id')

def test_update_dataframe_unknown_ids():
    df1 = pl.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
    df2 = pl.DataFrame({'id': [3], 'name': ['Charlie']})
    
    with pytest.raises(ValueError, match='Updates contains id_column id values not found in current data: \\[3\\]'):
        _update_dataframe(df1, df2, 'id')

def test_update_dataframe_empty_dataframes():
    df1 = pl.DataFrame({'id': [], 'name': []})
    df2 = pl.DataFrame({'id': [], 'name': []})
    
    result = _update_dataframe(df1, df2, 'id')
    assert result.shape == (0, 2)  # Should have 0 rows and 2 columns

def test_update_dataframe_single_row_update():
    df1 = pl.DataFrame({'id': [1], 'name': ['Alice']})
    df2 = pl.DataFrame({'id': [1], 'name': ['Alice Updated']})
    
    result = _update_dataframe(df1, df2, 'id')
    assert result.shape == (1, 2)
    assert result['name'].item() == 'Alice Updated'

def test_update_dataframe_preserve_existing_columns(base_dataframe):
    updates = pl.DataFrame({
        'id': [1],
        'name': ['Alice Updated']
    })
    
    result = _update_dataframe(base_dataframe, updates, 'id')
    assert 'age' in result.columns  # Original column should be preserved
    assert result.filter(pl.col('id') == 1)['age'].item() == 25  # Original value preserved

@pytest.fixture
def temp_project_dir(tmp_path):
    return tmp_path

@pytest.fixture
def project(temp_project_dir):
    return KPMSProject(temp_project_dir)

def test_kpms_project_init(temp_project_dir):
    project = KPMSProject(temp_project_dir)
    assert project.project_dir_path == temp_project_dir
    assert project.recordings_table_path == temp_project_dir / 'recordings.csv'

def test_get_recordings_not_exists(project):
    with pytest.raises(RuntimeError, match='does not exist'):
        project.get_recordings()

def test_add_recordings_new_file(project):
    new_recordings = pl.DataFrame({
        'name': ['rec1', 'rec2'],
        'value': ['val1', 'val2']
    })
    project.add_recordings(new_recordings)
    
    # Verify file was created and contains correct data
    assert project.recordings_table_path.exists()
    stored_recordings = pl.read_csv(project.recordings_table_path)
    assert stored_recordings.shape == (2, 2)
    assert stored_recordings['name'].to_list() == ['rec1', 'rec2']

def test_add_recordings_existing_file(project):
    # Create initial file
    initial_recordings = pl.DataFrame({
        'name': ['rec1'],
        'value': ['val1']
    })
    initial_recordings.write_csv(project.recordings_table_path)
    
    # Add new recordings
    new_recordings = pl.DataFrame({
        'name': ['rec2'],
        'value': ['val2']
    })
    project.add_recordings(new_recordings)
    
    # Verify combined data
    stored_recordings = pl.read_csv(project.recordings_table_path)
    assert stored_recordings.shape == (2, 2)
    assert stored_recordings['name'].to_list() == ['rec1', 'rec2']

def test_update_recordings(project):
    # Create initial file
    initial_recordings = pl.DataFrame({
        'name': ['rec1', 'rec2'],
        'value': ['val1', 'val2']
    })
    initial_recordings.write_csv(project.recordings_table_path)
    
    # Update one recording
    updates = pl.DataFrame({
        'name': ['rec1'],
        'value': ['val1_updated']
    })
    project.update_recordings(updates)
    
    # Verify updates
    stored_recordings = pl.read_csv(project.recordings_table_path)
    assert stored_recordings.shape == (2, 2)
    assert stored_recordings.filter(pl.col('name') == 'rec1')['value'].item() == 'val1_updated'
    assert stored_recordings.filter(pl.col('name') == 'rec2')['value'].item() == 'val2'

def test_update_recordings_not_exists(project):
    updates = pl.DataFrame({
        'name': ['rec1'],
        'value': ['val1']
    })
    with pytest.raises(RuntimeError, match='does not exist'):
        project.update_recordings(updates)

def test_add_recordings_invalid_data(project):
    # Try to add recordings with duplicate names
    invalid_recordings = pl.DataFrame({
        'name': ['rec1', 'rec1'],
        'value': ['val1', 'val2']
    })
    with pytest.raises(ValueError, match='Dataframe contains duplicate values in id_column name'):
        project.add_recordings(invalid_recordings)
