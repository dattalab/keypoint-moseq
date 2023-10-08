Advanced Topics
---------------

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

