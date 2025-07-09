.. _installing:

Installing QuScope
==================

QuScope is a Python package for quantum-enhanced electron microscopy and EELS analysis. It integrates Qiskit to provide advanced quantum image and spectral processing workflows.

Installation Options
--------------------

There are two recommended ways to install QuScope:

1. **Via GitHub (recommended for developers)**

   Clone the official repository and install in editable mode:

   .. code-block:: bash

      git clone https://github.com/rmsreis/quantum_algo_microscopy.git
      cd quantum_algo_microscopy
      pip install -e .

2. **Via `pip` (future release)**

   Once QuScope is published on PyPI:

   .. code-block:: bash

      pip install quscope

Environment Setup
-----------------

We recommend using a clean virtual environment:

.. code-block:: bash

   conda create -n quscope python=3.10
   conda activate quscope

Required Dependencies
---------------------

Minimum working versions:

- Python >= 3.10
- Qiskit >= 1.1.0
- NumPy >= 2.2.0
- SciPy >= 1.15.0
- matplotlib >= 3.10.0
- scikit-learn >= 1.7.0
- Pillow >= 11.2.0

To install them manually:

.. code-block:: bash

   pip install qiskit numpy scipy matplotlib scikit-learn pillow

To install all required packages at once:

.. code-block:: bash

   pip install -r requirements.txt

IBM Quantum Account Setup
-------------------------

QuScope supports integration with IBM Quantum simulators and real quantum devices via Qiskit.

To use IBM Quantum services:

1. Sign in or create an account at https://quantum.ibm.com.
2. Obtain your API token (from your account dashboard).
3. Set it in your environment:

   .. code-block:: bash

      export IBMQ_TOKEN='your-token-here'

   Or store it in a `.env` file and use Python’s `dotenv` package.

Jupyter Notebooks
-----------------

To run the full-featured notebooks:

.. code-block:: bash

   pip install notebook
   jupyter notebook notebooks/

The main notebook is:

- `notebooks/complete_quantum_microscopy_examples.ipynb`

Troubleshooting
---------------

If you encounter issues during installation:

- Ensure all system packages (e.g. `gcc`, `cmake`) are up-to-date
- Confirm compatibility with your Qiskit and NumPy versions
- Check that `pip` and `setuptools` are updated:

  .. code-block:: bash

     pip install --upgrade pip setuptools

For help, open an issue at:
https://github.com/rmsreis/quantum_algo_microscopy/issues