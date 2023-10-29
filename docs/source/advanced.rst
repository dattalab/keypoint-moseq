Advanced Topics
---------------

Reconstruct coordinates
~~~~~~~~~~~~~~~~~~~~~~~

During fitting, keypoint-MoSeq tries to estimate the "true" pose trajectory of the animal, discounting anomolous or low-confidence keypoints. The pose trajectory is stored in the model as a variable "x" that encodes a low-dimensional representation of the keypoints (similar to PCA). The code below shows how to project the pose trajectory back into the original coordinate space. This is useful for visualizing the estimated pose trajectory.::

    import os
    import h5py
    import numpy as np
    import jax.numpy as jnp
    from jax_moseq.utils import unbatch
    from jax_moseq.models.keypoint_slds import estimate_coordinates

    # load the model (change project_dir and model_name as needed)
    project_dir = 'demo_project'
    model_name = '2023_08_01-10_16_25'
    model, _, metadata, _ = kpms.load_checkpoint(project_dir, model_name)

    # compute the estimated coordinates
    Y_est = estimate_coordinates(
        jnp.array(model['states']['x']),
        jnp.array(model['states']['v']),
        jnp.array(model['states']['h']),
        jnp.array(model['params']['Cd'])
    )

    # generate a dictionary with reconstructed coordinates for each recording
    coordinates_est = unbatch(Y_est, *metadata)


The following code generates a video showing frames 0-3600 from one recording with the reconstructed keypoints overlaid.::

    config = lambda: kpms.load_config(project_dir)
    keypoint_data_path = 'dlc_project/videos' # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut')

    recording_name = '21_11_8_one_mouse.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000'
    video_path = 'dlc_project/videos/21_11_8_one_mouse.top.ir.mp4'

    output_path = os.path.splitext(video_path)[0]+'.reconstructed_keypoints.mp4'
    start_frame, end_frame = 0, 3600

    kpms.overlay_keypoints_on_video(
        video_path,
        coordinates_est[recording_name],
        skeleton = config()['skeleton'],
        bodyparts = config()['use_bodyparts'],
        output_path = output_path,
        frames = range(start_frame, end_frame)
    )



Automatic kappa scan
~~~~~~~~~~~~~~~~~~~~

Keypoint-MoSeq includes a hyperparameter called ``kappa`` that determines the rate of transitions between syllables. Higher values of kappa lead to longer syllables and smaller values lead to shorter syllables. Users should choose a value of kappa based their desired distribution of syllable durations. The code below shows how to automatically scan over a range of kappa values and choose te optimal value.::

    import numpy as np

    kappas = np.logspace(3,7,5)
    prefix = 'my_kappa_scan'

    for kappa in kappas:
        print(f"Fitting model with kappa={kappa}")
        model = kpms.update_hypparams(model, kappa=kappa)
        model_name = f'{prefix}-{kappa}'
        
        kpms.fit_model(
            model, data, metadata, project_dir,
            model_name, ar_only=True, num_iters=100, 
            save_every_n_iters=25);

    plot_kappa_scan(kappas, project_dir, model_name)


.. image:: _static/kappa_scan.jpg
   :align: center


Testing model convergence
~~~~~~~~~~~~~~~~~~~~~~~~~

