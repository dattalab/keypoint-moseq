import polars as pl
import numpy as np
from keypoint_moseq.project.kpms_project import KPMSProject
from keypoint_moseq.display.jupyter_display import JupyterDisplay

class Controller:
    def __init__(self, project: KPMSProject, display: JupyterDisplay):
        """Initialize the Controller that manages kpms activities

        Parameters
        ----------
        project : KPMSProject
            The 'model' in MVC that manages CRUD operations for kpms
        display : JupyterDisplay
            The 'view' in MVC that manages information display and user input for kpms
        """
        self.project: KPMSProject = project
        self.display: JupyterDisplay = display

    def set_group_labels(self):
        """Run the group labeling widget to assign experimental group labels to each recording
        in the project.
        """
        recordings: pl.DataFrame = self.project.get_recordings()
        
        if 'group' not in recordings.columns:
            recordings = recordings.with_columns(pl.lit('').alias('group'))

        recordings = recordings.sort(['group', 'name'])
            
        recording_names: np.ndarray[str] = recordings.get_column('name').to_numpy()
        initial_recording_groups: np.ndarray[str] = recordings.get_column('group').to_numpy()
        initial_group_labels = dict(zip(recording_names, initial_recording_groups))

        self.display.start_set_group_labels(initial_group_labels, self._save_group_labels)

    def _save_group_labels(self, group_labels: dict[str, str]):
        """Save the new expermental group labels for each recording in the project.

        Parameters
        ----------
        group_labels: dict[str, str]
            A mapping from recording names to experimental group labels.
        """
        recordings = pl.DataFrame({
            'name': list(group_labels.keys()),
            'group': list(group_labels.values())
        })

        self.project.update_recordings(recordings)