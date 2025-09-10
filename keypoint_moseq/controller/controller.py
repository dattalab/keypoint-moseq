import tqdm
import polars as pl
import keypoint_moseq as kpms
import numpy as np
import yaml
import logging
import sys
import h5py
from typing import Optional
from os import PathLike
from pathlib import Path
from importlib.resources import files
from textwrap import fill
from keypoint_moseq.io import _deeplabcut_loader, _sleap_loader, _anipose_loader, _sleap_anipose_loader, _nwb_loader, _facemap_loader, _freipose_loader, _dannce_loader, _name_from_path
from keypoint_moseq.project.kpms_project import KPMSProject
from keypoint_moseq.view.jupyter_display import JupyterDisplay

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

        _setup_logging(self.project.log_dir_path)
        sys.excepthook = _logging_excepthook
        logging.info('Controller initialized.')

    def set_group_labels(self):
        """Run the group labeling widget to assign experimental group labels to each recording
        in the project.
        """
        recordings: pl.DataFrame = self.project.get_recordings()
        logging.debug(f'Loaded recordings for setting group labels: {recordings}.')
        
        if 'group' not in recordings.columns:
            recordings = recordings.with_columns(pl.lit('').alias('group'))

        recordings = recordings.sort(['group', 'name'], descending=False)
            
        recording_names: np.ndarray[str] = recordings.get_column('name').to_numpy()
        initial_recording_groups: np.ndarray[str] = recordings.get_column('group').to_numpy()
        initial_group_labels = dict(zip(recording_names, initial_recording_groups))

        logging.info('Launching set group labels widget.')
        self.display.start_set_group_labels(initial_group_labels, self._save_group_labels)

    def _save_group_labels(self, group_labels: dict[str, str]):
        """Save the new expermental group labels for each recording in the project.

        Parameters
        ----------
        group_labels: dict[str, str]
            A mapping from recording names to experimental group labels.
        """
        logging.info('Saving group labels.')
        recordings = pl.DataFrame({
            'name': list(group_labels.keys()),
            'group': list(group_labels.values())
        })
        self.project.update_recordings(recordings)

def prepare_for_analysis(project_dir: PathLike):
    project_dir = Path(project_dir)
    recordings_path = project_dir / 'recordings.csv'
    if recordings_path.exists():
        return

    results_h5s = list(project_dir.glob('*/results.h5'))
    
    unique_session_names = set()
    
    for h5_path in results_h5s:
        with h5py.File(h5_path, 'r') as f:
            unique_session_names.update(f.keys())
    
    recordings = pl.DataFrame({
        'name': sorted(list(unique_session_names))
    })
    
    recordings.write_csv(recordings_path)

_CONTROLLER: Optional[Controller] = None

def _setup_logging(log_dir_path: PathLike):
    if logging.getLogger().handlers:
        return # Already configured
    
    logging_config_path = files('keypoint_moseq') / 'logging_config.yaml'
    with open(logging_config_path) as f:
        logging_config = yaml.safe_load(f)

    logging_config['handlers']['file']['filename'] = str(log_dir_path / 'log.log')
    logging.config.dictConfig(logging_config)

def _logging_excepthook(exctype, value, tb):
    logging.getLogger().critical(
        'unhandled exception',
        exc_info=(exctype, value, tb)
    )

def initialize_analysis_notebook(project_dir: str):
    global _CONTROLLER
    project = kpms.KPMSProject(project_dir)
    disp = kpms.JupyterDisplay()
    _CONTROLLER = kpms.Controller(project, disp)
    prepare_for_analysis(project_dir)

def set_group_labels():
    _CONTROLLER.set_group_labels()
