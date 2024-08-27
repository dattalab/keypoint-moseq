Keypoint MoSeq
==============


.. list-table::
   :widths: 30 30 30 30 30
   :header-rows: 0

   * - `GitHub <https://github.com/dattalab/keypoint-moseq/>`_
     - `Colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>`_
     - `Paper <https://www.nature.com/articles/s41592-024-02318-2>`_
     - `Slack <https://join.slack.com/t/moseqworkspace/shared_invite/zt-151x0shoi-z4J0_g_5rwJDlO1IfCU34A>`_
     - `License <https://github.com/dattalab/keypoint-moseq/blob/main/LICENSE.md>`_

.. image:: _static/logo.jpg
   :align: center

Motion Sequencing (MoSeq) is an unsupervised machine learning method for animal behavior analysis. Given behavioral recordings, MoSeq learns a set of stereotyped movement patterns and when they occur over time. This package provides tools for fitting a MoSeq model to keypoint tracking data and analyzing the results.


.. toctree::
   :caption: Setup
   
   install
   Google colab <https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb>


.. toctree::
   :caption: Tutorials

   modeling
   analysis

.. toctree::
   :caption: FAQs

   FAQs

.. toctree::
   :caption: Advanced topics

   advanced

.. toctree::
   :caption: Developer API

   fitting
   viz
   io
   util
   calibration
