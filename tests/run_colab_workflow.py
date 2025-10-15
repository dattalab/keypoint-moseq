"""
Adapted version of colab notebook for local execution with DLC example data
This script runs with reduced iterations for testing purposes
"""

import os
import time
import tempfile
import keypoint_moseq as kpms
import numpy as np

# Track execution time
start_time = time.time()

# Setup paths
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo_root, "docs", "source", "dlc_example_project")
dlc_config_path = os.path.join(data_dir, "config.yaml")
videos_dir = os.path.join(data_dir, "videos")

# Create temporary project directory
project_dir = tempfile.mkdtemp(prefix="kpms_test_")
print(f"Project directory: {project_dir}")
print(f"Data directory: {data_dir}")

# Create config lambda
config = lambda: kpms.load_config(project_dir)

print("\n=== Step 1: Setup Project ===")
step_start = time.time()
kpms.setup_project(project_dir, deeplabcut_config=dlc_config_path, overwrite=True)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 2: Update Config ===")
step_start = time.time()
kpms.update_config(
    project_dir,
    video_dir=videos_dir,
    anterior_bodyparts=["nose"],
    posterior_bodyparts=["spine4"],
    use_bodyparts=[
        "spine4",
        "spine3",
        "spine2",
        "spine1",
        "head",
        "nose",
        "right ear",
        "left ear",
    ],
    fps=30,
)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 3: Load Keypoints ===")
step_start = time.time()
coordinates, confidences, bodyparts = kpms.load_keypoints(videos_dir, "deeplabcut")
print(f"Loaded {len(coordinates)} recordings")
print(f"Bodyparts: {bodyparts}")
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 4: Format Data ===")
step_start = time.time()
data, metadata = kpms.format_data(coordinates, confidences, **config())
print(f"Formatted {len(metadata)} recordings")
print(f"Data keys: {list(data.keys())}")
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 5: Outlier Removal ===")
step_start = time.time()
kpms.update_config(project_dir, outlier_scale_factor=6.0)
coordinates, confidences = kpms.outlier_removal(
    coordinates,
    confidences,
    project_dir,
    overwrite=True,  # Force overwrite for testing
    **config(),
)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 6: Reformat Data After Outlier Removal ===")
step_start = time.time()
data, metadata = kpms.format_data(coordinates, confidences, **config())
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 7: Skip Calibration (Interactive Widget) ===")
print("Skipping noise_calibration() - requires manual interaction")

print("\n=== Step 8: Fit PCA ===")
step_start = time.time()
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)
kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 9: Update Latent Dimensions ===")
step_start = time.time()
kpms.update_config(project_dir, latent_dim=4)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 10: Estimate Hyperparameters ===")
step_start = time.time()
kpms.update_config(
    project_dir,
    sigmasq_loc=kpms.estimate_sigmasq_loc(
        data["Y"], data["mask"], filter_size=config()["fps"]
    ),
)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 11: Initialize Model ===")
step_start = time.time()
model = kpms.init_model(data, pca=pca, **config())
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 12: Fit AR-HMM (Reduced Iterations) ===")
step_start = time.time()
num_ar_iters = 10  # Reduced from 50 for testing
print(f"Running {num_ar_iters} iterations...")
model, model_name = kpms.fit_model(
    model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
)
print(f"Model name: {model_name}")
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 13: Fit Full Model (Reduced Iterations) ===")
step_start = time.time()
# Load checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)
# Update kappa
model = kpms.update_hypparams(model, kappa=1e4)
# Fit with reduced iterations
num_full_iters = 20  # Reduced from 500 for testing
print(f"Running {num_full_iters} additional iterations...")
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + num_full_iters,
)[0]
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 14: Reindex Syllables ===")
step_start = time.time()
kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 15: Extract Results ===")
step_start = time.time()
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
results = kpms.extract_results(model, metadata, project_dir, model_name)
print(f"Extracted results for {len(results)} recordings")
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 16: Save Results as CSV ===")
step_start = time.time()
kpms.save_results_as_csv(results, project_dir, model_name)
print(f"Time: {time.time() - step_start:.2f}s")

print("\n=== Step 17: Generate Visualizations ===")
step_start = time.time()
results = kpms.load_results(project_dir, model_name)

# Trajectory plots
kpms.generate_trajectory_plots(
    coordinates, results, project_dir, model_name, **config()
)

# Grid movies
kpms.generate_grid_movies(
    results, project_dir, model_name, coordinates=coordinates, **config()
)

# Dendrogram
kpms.plot_similarity_dendrogram(
    coordinates, results, project_dir, model_name, **config()
)
print(f"Time: {time.time() - step_start:.2f}s")

# Final summary
total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"WORKFLOW COMPLETED SUCCESSFULLY")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Project directory: {project_dir}")
print(f"Model name: {model_name}")
print("=" * 60)

# List generated files
print("\nGenerated files:")
for root, dirs, files in os.walk(project_dir):
    level = root.replace(project_dir, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:10]:  # Limit to first 10 files per directory
        print(f"{subindent}{file}")
    if len(files) > 10:
        print(f"{subindent}... and {len(files) - 10} more files")
