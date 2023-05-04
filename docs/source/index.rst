Keypoint MoSeq
==============


.. list-table::
   :widths: 30 30 30 30 30
   :header-rows: 0

   * - `GitHub <https://github.com/dattalab/keypoint-moseq/>`_
     - `Colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>`_
     - `Paper <https://www.biorxiv.org/content/10.1101/2023.03.16.532307v2>`_
     - `Slack <https://join.slack.com/t/moseqworkspace/shared_invite/zt-151x0shoi-z4J0_g_5rwJDlO1IfCU34A>`_
     - `License <https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md>`_


.. note::
    We detected a bug in versions 0.0.4 and 0.0.5 of keypoint-moseq that led to incorrect model outputs. **This affects you if you installed/updated the code between April 25 and May 5.** The bug has been fixed in version 0.1.0. You can update using ``pip install -U keypoint-moseq``. Note that after updating, the "Apply model" step of the original tutorial will no longer work and must be replaced with the `"Extract model results" step in the updated tutorial <https://keypoint-moseq.readthedocs.io/en/latest/tutorial.html#extract-model-results>`_.

.. image:: _static/logo.jpg
   :align: center


Motion Sequencing (MoSeq) is an unsupervised machine learning method for animal behavior analysis. Given behavioral recordings, MoSeq learns a set of stereotyped movement patterns and when they occur over time. This package provides tools for fitting a MoSeq model to keypoint tracking data. 


.. toctree::
   :caption: Setup
   
   install
   Google colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>


.. toctree::
   :caption: Tutorial

   tutorial
   changepoints

.. toctree::
   :caption: FAQs

   FAQs

.. toctree::
   :caption: Developer API

   fitting
   viz
   io
   util
   calibration
