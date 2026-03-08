Installation
============

Install from GitHub
-------------------

.. code-block:: bash

   pip install git+https://github.com/zou-group/CORAL

Conda Environment
-----------------

A full conda environment specification is provided in ``environment.yml``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate coral_env

Key Dependencies
----------------

- **PyTorch** and **PyTorch Geometric** -- deep learning and graph neural networks
- **scanpy** / **anndata** -- single-cell data structures and analysis
- **scikit-learn** -- clustering and evaluation metrics
- **OpenCV** -- image alignment (BRISK, VGG, RANSAC)
- **matplotlib** / **seaborn** -- visualization
