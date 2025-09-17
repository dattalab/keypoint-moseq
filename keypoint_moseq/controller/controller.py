import tqdm
import polars as pl
import keypoint_moseq as kpms
import numpy as np
import yaml
import logging
import sys
import h5py
from scipy import stats
from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd
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

    def plot_syllable_usage(self, model_name: str):
        frames: pl.LazyFrame = self.project.get_frames()
        recordings: pl.LazyFrame = self.project.get_recordings().lazy()

        frames = frames.filter(pl.col('model').eq(model_name))
        df = frames.join(recordings, left_on='session_name', right_on='name', how='full')
        usages = (
            df
            .with_columns(
                pl.len().over('session_name').alias('instances_in_session'))
            .group_by('syllable', 'session_name', 'group', 'instances_in_session')
            .agg(pl.len().alias('syllable_instances_in_session'))
            .with_columns(pl.col('syllable_instances_in_session') / pl.col('instances_in_session'))
            .collect()
            .to_numpy().T
        )

        stats = _group_syllable_comparison(usages[0], usages[2], usages[4])
        
        self.display.display_group_syllable_differences_plot(
            stats['centers'],
            stats['errors'],
            stats['significant_comparisons'],
            stats['group_labels'],
            stats['syllables'],
            r'Usage (% frames)'
        )

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

    def label_syllables(self, model_name: str):
        """Run the syllable labeling widget to assign qualitative labels to each syllable
        for the specified model.
        
        Parameters
        ----------
        model_name : str
            The name of the model to label syllables for
        """
        syllables: pl.DataFrame = self.project.get_syllables()
        logging.debug(f'Loaded syllables for labeling: {syllables}.')
        
        # Filter by model name
        model_syllables = syllables.filter(pl.col('model') == model_name)
        
        if len(model_syllables) == 0:
            raise ValueError(f"No syllables found for model '{model_name}'")
        
        # Validate required columns
        required_columns = ['syllable_id', 'syllable', 'grid_movie_path']
        missing_columns = [col for col in required_columns if col not in model_syllables.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add missing optional columns with defaults
        if 'label' not in model_syllables.columns:
            model_syllables = model_syllables.with_columns(pl.lit('').alias('label'))
        if 'short_description' not in model_syllables.columns:
            model_syllables = model_syllables.with_columns(pl.lit('').alias('short_description'))
        
        # Sort by syllable number
        model_syllables = model_syllables.sort('syllable')
        
        # Convert to list of dicts format expected by widget
        syllables_list = []
        for row in model_syllables.iter_rows(named=True):
            # Convert relative grid movie path to absolute path for video player
            relative_path = row['grid_movie_path']
            absolute_path = str(self.project.project_dir_path / relative_path) if relative_path else ''
            
            syllables_list.append({
                'syllable_id': row['syllable_id'],  # Hidden from user but flows through
                'syllable': row['syllable'],
                'grid_movie_path': absolute_path,  # Convert to absolute path for video player
                'label': row['label'],
                'short_description': row['short_description']  # Widget now expects underscores
            })
        
        logging.info(f'Launching syllable labeling widget for model {model_name}.')
        self.display.start_label_syllables(syllables_list, self._save_syllable_info)

    def _save_syllable_info(self, syllable_info: list[dict]):
        """Save the new syllable labels and descriptions for the project.

        Parameters
        ----------
        syllable_info: list[dict]
            A list of dictionaries containing syllable information with keys:
            'syllable_id', 'label', 'short_description'
        """
        logging.info('Saving syllable info.')
        syllable_updates = pl.DataFrame([
            {
                'syllable_id': info['syllable_id'],
                'label': info['label'],
                'short_description': info['short_description']
            }
            for info in syllable_info
        ])
        self.project.update_syllables(syllable_updates)

def _is_results_h5_file(h5_path: Path) -> bool:
    """Check if an H5 file has the expected structure of a results file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if file has session-level groups
            if not f.keys():
                return False
            
            # Check a few sessions to validate structure
            session_names = list(f.keys())
            sessions_to_check = session_names[:min(3, len(session_names))]  # Check up to 3 sessions
            
            for session_name in sessions_to_check:
                if session_name not in f:
                    return False
                    
                session_group = f[session_name]
                if not hasattr(session_group, 'keys'):  # Not a group
                    return False
                
                # Check for required datasets
                required_datasets = ['syllable', 'centroid', 'heading', 'latent_state']
                for dataset_name in required_datasets:
                    if dataset_name not in session_group:
                        return False
                
                # Basic validation of data shapes/types
                syllable_data = session_group['syllable']
                centroid_data = session_group['centroid'] 
                
                # Check that syllable is 1D and centroid is 2D with matching length
                if (len(syllable_data.shape) != 1 or 
                    len(centroid_data.shape) != 2 or 
                    syllable_data.shape[0] != centroid_data.shape[0]):
                    return False
                
                # Check centroid has 2 columns (x, y)
                if centroid_data.shape[1] != 2:
                    return False
                
                # Ensure data is not empty
                if syllable_data.shape[0] == 0:
                    return False
            
            return True
            
    except Exception:
        return False

def _group_syllable_comparison(syllables: np.ndarray, groups: np.ndarray, values: np.ndarray):
    """Compare syllable usage between groups using Kruskal-Wallis and Dunn's post-hoc tests.
    
    For each syllable, performs Kruskal-Wallis test across groups. If significant, 
    runs Dunn's post-hoc test with Bonferroni correction for pairwise comparisons.
    
    Parameters
    ----------
    syllables : np.ndarray
        Array of syllable IDs for each data point
    groups : np.ndarray  
        Array of group labels for each data point
    values : np.ndarray
        Array of values for each data point
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'centers': means with shape (n_syllables, n_groups)
        - 'errors': SEMs with shape (n_syllables, n_groups) 
        - 'significant_comparisons': list of lists of significant group pair tuples
        - 'group_labels': array of unique group labels, ordered to the columns of centers
        - 'syllables': array of unique syllables, ordered to the rows of centers
    """
    # Get unique syllables and groups
    unique_syllables = np.unique(syllables)
    unique_groups = np.unique(groups)
    
    n_syllables = len(unique_syllables)
    n_groups = len(unique_groups)
    
    # Initialize output arrays
    centers = np.full((n_syllables, n_groups), np.nan)
    errors = np.full((n_syllables, n_groups), np.nan)
    significant_comparisons = []
    
    # For each syllable, perform statistical tests
    for syll_idx, syllable in enumerate(unique_syllables):
        syllable_mask = syllables == syllable
        syllable_values = values[syllable_mask]
        syllable_groups = groups[syllable_mask]
        
        # Calculate means and SEMs for each group
        group_data = []
        for group_idx, group in enumerate(unique_groups):
            group_mask = syllable_groups == group
            group_values = syllable_values[group_mask]
            
            if len(group_values) > 0:
                centers[syll_idx, group_idx] = np.mean(group_values)
                errors[syll_idx, group_idx] = stats.sem(group_values)
                group_data.append(group_values)
            else:
                group_data.append(np.array([]))
        
        # Only test if we have data for at least 2 groups with non-empty data
        valid_groups = [i for i, data in enumerate(group_data) if len(data) > 0]
        syllable_comparisons = []
        
        if len(valid_groups) >= 2:
            # Run Kruskal-Wallis test
            valid_group_data = [group_data[i] for i in valid_groups]
            kw_stat, kw_pval = kruskal(*valid_group_data)
            
            # If Kruskal-Wallis is significant, run Dunn's post-hoc test
            if kw_pval < 0.05:
                # Prepare data for Dunn's test
                dunn_data = []
                dunn_groups = []
                
                for i, group_idx in enumerate(valid_groups):
                    group_values = valid_group_data[i]
                    dunn_data.extend(group_values)
                    dunn_groups.extend([unique_groups[group_idx]] * len(group_values))
                
                # Create DataFrame for scikit-posthocs
                dunn_df = pd.DataFrame({
                    'value': dunn_data,
                    'group': dunn_groups
                })
                
                # Run Dunn's test with Bonferroni correction
                dunn_results = sp.posthoc_dunn(
                    dunn_df, val_col='value', group_col='group', p_adjust='bonferroni'
                )
                
                # Extract significant comparisons
                for i in range(len(valid_groups)):
                    for j in range(i + 1, len(valid_groups)):
                        group1_name = unique_groups[valid_groups[i]]
                        group2_name = unique_groups[valid_groups[j]]
                        
                        if (group1_name in dunn_results.index and 
                            group2_name in dunn_results.columns and
                            dunn_results.loc[group1_name, group2_name] < 0.05):
                            syllable_comparisons.append((valid_groups[i], valid_groups[j]))
        
        significant_comparisons.append(syllable_comparisons)
    
    return {
        'centers': centers,
        'errors': errors, 
        'significant_comparisons': significant_comparisons,
        'group_labels': unique_groups,
        'syllables': unique_syllables
    }


def prepare_for_analysis(project_dir: PathLike):
    project_dir = Path(project_dir)
    tables_dir = project_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    recordings_path = tables_dir / 'recordings.csv'
    frames_path = tables_dir / 'frames.parquet'
    instances_path = tables_dir / 'instances.parquet'
    syllables_path = tables_dir / 'syllables.csv'
    
    # First, discover all H5 files recursively and validate them
    print("Scanning for results H5 files...")
    all_h5_files = list(project_dir.rglob('*.h5'))
    
    # Validate which H5 files are actually results files
    results_h5s = []
    for h5_path in all_h5_files:
        if _is_results_h5_file(h5_path):
            results_h5s.append(h5_path)
    
    if not results_h5s:
        print("No valid results H5 files found - nothing to prepare")
        return
        
    print(f"Found {len(results_h5s)} valid model result files")
    
    # Extract all current session names from H5 files
    current_session_names = set()
    for h5_path in results_h5s:
        with h5py.File(h5_path, 'r') as f:
            current_session_names.update(f.keys())
    
    print(f"Found {len(current_session_names)} total sessions across all models")
    
    # Determine if we need to regenerate tables
    needs_regeneration = False
    
    # Check if recordings.csv exists and compare sessions
    if recordings_path.exists():
        existing_recordings = pl.read_csv(recordings_path)
        existing_session_names = set(existing_recordings['name'].to_list())
        new_sessions = current_session_names - existing_session_names
        
        if new_sessions:
            print(f"Found {len(new_sessions)} new sessions: {sorted(new_sessions)}")
            needs_regeneration = True
    else:
        needs_regeneration = True
    
    # Check if other tables are missing
    missing_tables = []
    if not frames_path.exists():
        missing_tables.append('frames.parquet')
    if not instances_path.exists():
        missing_tables.append('instances.parquet')
    if not syllables_path.exists():
        missing_tables.append('syllables.csv')
    
    if missing_tables:
        print(f"Missing tables: {missing_tables}")
        needs_regeneration = True
    
    # If no changes needed, exit early
    if not needs_regeneration:
        print("All tables are up to date - nothing to initialize")
        return
        
    print("Initializing analysis tables...")
    
    # Process all H5 files to extract data
    session_data = []
    syllable_data = []
    instance_id_offset = 0
    temp_files = []
    
    for h5_path in results_h5s:
        model_name = h5_path.parent.name
        
        print(f"Processing model '{model_name}' from {h5_path.relative_to(project_dir)}")
        model_syllables = set()
        
        with h5py.File(h5_path, 'r') as f:
            for session_name in f.keys():
                session_group = f[session_name]
                
                # Get frame count for recordings table
                num_frames = session_group['syllable'].shape[0]
                session_data.append({
                    'name': session_name,
                    'num_frames': num_frames
                })
                
                # Extract all frame-level data
                centroid = session_group['centroid'][:]
                heading = session_group['heading'][:]
                latent_state = session_group['latent_state'][:]
                syllable = session_group['syllable'][:]
                
                # Collect unique syllables for this model
                model_syllables.update(syllable)
                
                # Calculate behavioral onset and global instance_id for this session
                is_onset = np.zeros(num_frames, dtype=int)
                is_onset[1:] = (syllable[1:] != syllable[:-1]).astype(int)
                
                # Create global onset (syllable changes + session start)
                global_onset = is_onset.copy()
                global_onset[0] = 1  # First frame of session is always an onset
                
                # Calculate instance_id with global offset
                local_instance_id = np.cumsum(global_onset)
                instance_id = local_instance_id + instance_id_offset
                
                # Update offset for next session
                instance_id_offset = instance_id[-1]
                
                # Create DataFrame for this session's frames
                frame_data = {
                    'model': [model_name] * num_frames,
                    'session_name': [session_name] * num_frames,
                    'frame_num': range(num_frames),
                    'centroid_x': centroid[:, 0],
                    'centroid_y': centroid[:, 1], 
                    'heading': heading,
                    'syllable': syllable,
                    'is_onset': is_onset, 
                    'instance_id': instance_id
                }
                
                # Add latent state dimensions
                n_latent_dims = latent_state.shape[1]
                for dim in range(n_latent_dims):
                    frame_data[f'latent_state_{dim + 1}'] = latent_state[:, dim]
                
                session_frames = pl.DataFrame(frame_data)
                
                # Write this session to a temporary parquet file
                temp_file = frames_path.with_suffix(f'.temp_{len(temp_files)}.parquet')
                session_frames.write_parquet(temp_file)
                temp_files.append(temp_file)
        
        # Process syllables for this model
        grid_movies_dir = h5_path.parent / 'grid_movies'
        for syllable_num in model_syllables:
            grid_movie_file = grid_movies_dir / f'syllable{syllable_num}.mp4'
            # Store relative path from project directory, not absolute path
            if grid_movie_file.exists():
                grid_movie_path = str(grid_movie_file.relative_to(project_dir))
            else:
                grid_movie_path = ''
            syllable_data.append({
                'model': model_name,
                'syllable': syllable_num,
                'grid_movie_path': grid_movie_path
            })
    
    # Update recordings table - merge with existing data to preserve group labels
    new_recordings_df = pl.DataFrame(session_data).sort('name')
    
    if recordings_path.exists():
        # Load existing recordings and merge with new data
        existing_recordings = pl.read_csv(recordings_path)
        new_session_names = set(new_recordings_df['name'].to_list())
        existing_session_names = set(existing_recordings['name'].to_list())
        new_sessions = new_session_names - existing_session_names
        
        if new_sessions:
            # Filter new recordings to only those not already in existing
            new_recordings_only = new_recordings_df.filter(pl.col('name').is_in(list(new_sessions)))
            
            # Add any missing columns from existing recordings (e.g., 'group')
            for col in existing_recordings.columns:
                if col not in new_recordings_only.columns:
                    new_recordings_only = new_recordings_only.with_columns(pl.lit('').alias(col))
            
            # Combine existing and new recordings
            combined_recordings = pl.concat([existing_recordings, new_recordings_only]).sort('name')
            combined_recordings.write_csv(recordings_path)
            print(f"Updated recordings.csv: added {len(new_sessions)} new recordings")
    else:
        # No existing recordings, create new file
        new_recordings_df.write_csv(recordings_path)
        print(f"Created recordings.csv with {len(new_recordings_df)} recordings")
    
    # Create and save syllables table with global syllable_id
    syllables = pl.DataFrame(syllable_data).sort(['model', 'syllable'])
    syllables = syllables.with_columns(
        pl.int_range(1, len(syllables) + 1).alias('syllable_id')
    )
    syllables.write_csv(syllables_path)
    print(f"Created syllables.csv with {len(syllables)} syllables across all models")
    
    # Concatenate all temporary files
    lazy_frames = [pl.scan_parquet(temp_file) for temp_file in temp_files]
    combined_lazy = pl.concat(lazy_frames)
    combined_lazy.sink_parquet(frames_path)
    print(f"Created frames.parquet with frame-level data")
    
    # Clean up temporary files
    for temp_file in temp_files:
        temp_file.unlink()
    
    # Create instance-level summary table
    sample_frame = pl.scan_parquet(frames_path).select(pl.col("*")).head(1).collect()
    latent_cols = [col for col in sample_frame.columns if col.startswith('latent_state_')]
    
    agg_exprs = [
        pl.col('model').first(),
        pl.col('session_name').first(),
        pl.col('frame_num').min().alias('starting_frame_num'),
        pl.col('frame_num').count().alias('duration_frames'),
        pl.col('centroid_x').mean(),
        pl.col('centroid_y').mean(),
        pl.col('heading').mean(),
        pl.col('syllable').first(),
    ]
    
    for col in latent_cols:
        agg_exprs.append(pl.col(col).mean())
    
    instances_lazy = (
        pl.scan_parquet(frames_path)
        .group_by('instance_id')
        .agg(agg_exprs)
    )
    instances_lazy.sink_parquet(instances_path)
    print(f"Created instances.parquet with syllable instance summaries")
    
    print("Analysis tables initialization complete!")

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

def label_syllables(model_name: str):
    _CONTROLLER.label_syllables(model_name)

def plot_syllable_usage(model_name: str):
    _CONTROLLER.plot_syllable_usage(model_name)