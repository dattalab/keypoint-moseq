import polars as pl
import numpy as np
from pathlib import Path
from os import PathLike

def _check_id_column_integrity(dataframe: pl.DataFrame, id_column: str):
    """Check that a dataframe has an id column and no duplicates in it.

    Parameters
    ----------
    dataframe: pl.DataFrame
        The dataframe to check
    id_column: str
        The column to check for duplicates

    Raises
    ------
    ValueError
        Raised when dataframe does not have the id_column
    ValueError
        Raised when dataframe has duplicates in the id_column
    """
    if id_column not in dataframe.columns:
        raise ValueError(f'Dataframe must have column id_column "{id_column}". Got columns: {dataframe.columns}')

    ids: np.ndarray = dataframe.get_column(id_column).to_numpy()
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicated: np.ndarray = unique_ids[counts > 1]

    if len(duplicated) > 0:
        raise ValueError(f'Dataframe contains duplicate values in id_column {id_column}: {duplicated}')

def _update_dataframe(current_dataframe: pl.DataFrame, updates: pl.DataFrame, id_column: str) -> pl.DataFrame:
    """Update rows in current_dataframe with values from updates, matching on id_column.

    If current_dataframe has rows that don't exist in updates as defined by the values in id_column, those
    rows will stay as they are. The same goes for columns. If updates has columns that don't exist in current_dataframe,
    those columns will be added. Any rows for which those columns are not defined in updates will be given a null value
    for those columns. 

    Parameters
    ----------
    current_dataframe: pl.DataFrame
        The dataframe to update
    updates: pl.DataFrame
        The dataframe containing updated values
    id_column: str
        The column to match rows between dataframes

    Returns
    -------
    pl.DataFrame
        The updated dataframe

    Raises
    ------
    ValueError
        Raised when either dataframe does not have the id_column
    ValueError  
        Raised when either dataframe has duplicates in the id_column
    ValueError
        Raised when updates has values in id_column that don't exist in current_dataframe
    """
    _check_id_column_integrity(updates, id_column)
    _check_id_column_integrity(current_dataframe, id_column)

    new_ids: np.ndarray = updates.get_column(id_column).to_numpy()
    old_ids: np.ndarray = current_dataframe.get_column(id_column).to_numpy()
    unknown_ids = np.setdiff1d(new_ids, old_ids)

    if len(unknown_ids) > 0:
        raise ValueError(f'Updates contains id_column {id_column} values not found in current data: {unknown_ids}')

    id_indices = np.where(current_dataframe.get_column(id_column).is_in(new_ids))[0]
    updated_df = current_dataframe.clone()

    for column in updates.columns:
        if column not in updated_df.columns:
            null_value = pl.lit(None).cast(updates.get_column(column).dtype)
            updated_df = updated_df.with_columns(null_value.alias(column))

        updated_df[id_indices, column] = updates.get_column(column)

    return updated_df

class KPMSProject:
    def __init__(self, project_dir_path: PathLike):
        """Initializes the KPMSProject, which serves as the 'Model' in MVC for kpms.

        Defines project file structure through filepaths.

        Parameters
        ----------
        project_dir_path: PathLike
            The root path of the kpms project.
        """
        self.project_dir_path: Path = Path(project_dir_path)
        self.recordings_csv_path: Path = self.project_dir_path / 'recordings.csv'

    def get_recordings(self) -> pl.DataFrame:
        """Retrieve all recordings in the project. 

        Returns
        -------
        pl.DataFrame: Each row is a recording session in the project.

        Raises
        ------
        RuntimeError
            Raised when the recordings CSV does not exist.
        """
        if not self.recordings_csv_path.exists():
            raise RuntimeError(f'{self.recordings_csv_path} does not exist.')

        return pl.read_csv(self.recordings_csv_path)

    def add_recordings(self, new_recordings: pl.DataFrame):
        _check_id_column_integrity(new_recordings, 'name')

        if not self.recordings_csv_path.exists():
            new_recordings.write_csv(self.recordings_csv_path)
            return

        recordings = pl.read_csv(self.recordings_csv_path)
        recordings = pl.concat([recordings, new_recordings], how='diagonal_relaxed')

        recordings.write_csv(self.recordings_csv_path)

    def update_recordings(self, new_recordings: pl.DataFrame):
        """Update the project recordings CSV with new or updated features of each recording.

        Only the subset of recording names and features passed will be updated.

        Parameters
        ----------
        new_recordings: pl.DataFrame
            The updated recording information. Each row is a recording, and each column is a feature
            of the recordings.

        Raises
        ------
        RuntimeError
            Raised when the recordings CSV does not exist.
        ValueError
            Raised when new_recordings has invalid data (see update_dataframe)
        """
        if not self.recordings_csv_path.exists():
            raise RuntimeError(f'{self.recordings_csv_path} does not exist.')

        updated_recordings = _update_dataframe(
            self.get_recordings(),
            new_recordings,
            'name'
        )

        updated_recordings.write_csv(self.recordings_csv_path)