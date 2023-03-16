Keypoint MoSeq
==============

.. image:: logo.jpg
   :align: center

Motion Sequencing (MoSeq) is an unsupervised machine learning method for animal behavior analysis `(Wiltschko et al., 2015) <http://datta.hms.harvard.edu/wp-content/uploads/2018/01/pub_23.pdf>`_. Given behavioral recordings, MoSeq learns a set of stereotyped movement patterns and when they occur over time. This package provides tools for fitting a MoSeq model to keypoint tracking data. 

Links
-----

- `MoSeq website <https://dattalab.github.io/moseq2-website/index.html>`_

- `Keypoint-MoSeq paper <_static/BIORXIV-2023-532307v1-Datta.pdf>`_

- `Slack workspace <https://join.slack.com/t/moseqworkspace/shared_invite/zt-151x0shoi-z4J0_g_5rwJDlO1IfCU34A>`_


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Installation

   install

.. include:: install.rst

.. toctree::
   :caption: Tutorial

   tutorial

.. toctree::
   :caption: API Documentation

   fitting
   viz
   io
   util
   calibration
