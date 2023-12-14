Exporting pose estimates
------------------------

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
--------------------

Keypoint-MoSeq includes a hyperparameter called ``kappa`` that determines the rate of transitions between syllables. Higher values of kappa lead to longer syllables and smaller values lead to shorter syllables. Users should choose a value of kappa based their desired distribution of syllable durations. The code below shows how to automatically scan over a range of kappa values and choose te optimal value.::

    import numpy as np

    kappas = np.logspace(3,7,5)
    decrease_kappa_factor = 10
    num_ar_iters = 50
    num_full_iters = 200

    prefix = 'my_kappa_scan'

    for kappa in kappas:
        print(f"Fitting model with kappa={kappa}")
        model_name = f'{prefix}-{kappa}'
        model = kpms.init_model(data, pca=pca, **config())
        
        # stage 1: fit the model with AR only
        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model, 
            data, 
            metadata, 
            project_dir, 
            model_name, 
            ar_only=True, 
            num_iters=num_ar_iters, 
            save_every_n_iters=25
        )[0];

        # stage 2: fit the full model
        model = kpms.update_hypparams(model, kappa=kappa/decrease_kappa_factor)
        kpms.fit_model(
            model, 
            data, 
            metadata, 
            project_dir, 
            model_name, 
            ar_only=False, 
            start_iter=num_ar_iters,
            num_iters=num_full_iters, 
            save_every_n_iters=25
        );

    kpms.plot_kappa_scan(kappas, project_dir, prefix)


.. image:: _static/kappa_scan.jpg
   :align: center




Model selection and comparison
------------------------------

Keypoint-MoSeq uses a stochastic fitting procedure, and thus produces slightly different syllable segmentations when run multiple times with different random seeds. Below, we show how to fit multiple models, compare the resulting syllables, and then select an optimal model for further analysis. It may also be useful in some cases to show that downstream analyses are robust to the choice of model.


.. _fitting-multiple-models:

Fitting multiple models
~~~~~~~~~~~~~~~~~~~~~~~

The code below shows how to fit multiple models with different random seeds.::

    import jax

    num_model_fits = 20
    prefix = 'my_models'

    ar_only_kappa = 1e6
    num_ar_iters = 50

    full_model_kappa = 1e4
    num_full_iters = 500

    for restart in range(num_model_fits):
        print(f"Fitting model {restart}")
        model_name = f'{prefix}-{restart}'
        
        model = kpms.init_model(
            data, pca=pca, **config(), seed=jax.random.PRNGKey(restart)
        )

        # stage 1: fit the model with AR only
        model = kpms.update_hypparams(model, kappa=ar_only_kappa)
        model = kpms.fit_model(
            model,
            data, 
            metadata, 
            project_dir, 
            model_name,
            ar_only=True, 
            num_iters=num_ar_iters
        )[0]

        # stage 2: fit the full model
        model = kpms.update_hypparams(model, kappa=full_model_kappa)
        kpms.fit_model(
            model, 
            data, 
            metadata, 
            project_dir, 
            model_name,
            ar_only=False, 
            start_iter=num_ar_iters,
            num_iters=num_full_iters
        );

        kpms.reindex_syllables_in_checkpoint(project_dir, model_name);
        model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
        results = kpms.extract_results(model, metadata, project_dir, model_name)
        
        

Comparing syllables
~~~~~~~~~~~~~~~~~~~

To get a sense of the variability across model runs, it may be useful to compare syllables produced by each model. The code below shows how to load results from two models runs (e.g., produced by the code above) and plot a confusion matrix showing the overlap between syllable labels.::

    model_name_1 = 'my_models-0'
    model_name_2 = 'my_models-1'

    results_1 = kpms.load_results(project_dir, model_name_1)
    results_2 = kpms.load_results(project_dir, model_name_2)

    kpms.plot_confusion_matrix(results_1, results_2);


.. image:: _static/confusion_matrix.jpg


Selecting a model
~~~~~~~~~~~~~~~~~

We developed a matric called the expected marginal likelihood (EML) score that can be used to rank models. To calculate EML scores, you must first fit an ensemble of models to a given dataset, as shown in :ref:`Fitting multiple models <fitting-multiple-models>`. The code below loads this ensemble and then calculates the EML score for each model. The model with the highest EML score can then be selected for further analysis.::


    # change the following line as needed
    model_names = ['my_models-{}'.format(i) for i in range(20)]

    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model_name}")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)


.. image:: _static/EML_scores.jpg


Model averaging
~~~~~~~~~~~~~~~

Keypoint-MoSeq is probabilistic. So even once fitting is complete and the syllable parameters are fixed, there is still a distribution of possible syllable sequences given the observed data. In the default pipeline, one such sequence is sampled from this distribution and used for downstream analyses. Alternatively, one can estimate the marginal probability distribution over syllable labels at each timepoint. The code below shows how to do this. It can be applied to new data or the same data that was used for fitting (or a combination of the two).::

    burnin_iters = 50
    num_samples = 100
    steps_per_sample = 5

    # load the model (change `project_dir` and `model_name` as needed)
    model = kpms.load_checkpoint(project_dir, model_name)[0]

    # load data (e.g. from deeplabcut)
    data_path = 'path/to/data/' # can be a file, a directory, or a list of files
    coordinates, confidences, bodyparts = kpms.load_keypoints(data_path, 'deeplabcut')
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    # compute the marginal probabilities of syllable labels
    marginal_probs = kpms.estimate_syllable_marginals(
        model, data, metadata, burnin_iters, num_samples, steps_per_sample, **config()
    )