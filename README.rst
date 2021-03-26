WATCH
=====

|master-pipeline| |master-coverage|

This repository addresses the algorithmic challenges of the IARPA SMART
program. The goal of this software is analyze space-based imagery to
perform broad-area search for natural and anthropogenic events and
characterize their extent and progression in time and space.


The following table provides links to relevant resources for the SMART WATCH project:

+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH Python Module | https://gitlab.kitware.com/smart/watch/                        |
+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH DVC Repo      | https://gitlab.kitware.com/smart/smart_watch_dvc/              |
+-------------------------------+----------------------------------------------------------------+
| The SMART WATCH Collection    | https://data.kitware.com/#collection/602457272fa25629b95d1718  |
+-------------------------------+----------------------------------------------------------------+

Getting Started
---------------

Install Conda
~~~~~~~~~~~~~

Conda3 is required to set up the environment using Python 3. Follow the
instructions below to install miniconda3 for Linux. For Windows 10
users, the Windows Subsystem for Linux (WSL) allows you to run Linux
within Windows.

.. code:: bash

    # Download the conda install script into a temporary directory
    mkdir -p ~/tmp
    cd ~/tmp

    # To update to a newer version see:
    # https://docs.conda.io/en/latest/miniconda_hashes.html for updating
    CONDA_INSTALL_SCRIPT=Miniconda3-py38_4.9.2-Linux-x86_64.sh
    CONDA_EXPECTED_SHA256=1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
    curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT
    CONDA_GOT_SHA256=$(sha256sum $CONDA_INSTALL_SCRIPT | cut -d' ' -f1)
    # For security, it is important to verify the hash
    if [[ "$CONDA_GOT_SHA256" != "$CONDA_EXPECTED_SHA256_HASH" ]]; then
        echo "Downloaded file does not match hash! DO NOT CONTINUE!"
        exit 1;
    fi
    chmod +x $CONDA_INSTALL_SCRIPT 

    # Install miniconda to user local directory
    _CONDA_ROOT=$HOME/.local/conda
    sh $CONDA_INSTALL_SCRIPT -b -p $_CONDA_ROOT
    # Activate the basic conda environment
    source $_CONDA_ROOT/etc/profile.d/conda.sh
    # Update the base and create a virtual environment named py38
    conda update --name base conda --yes 


Create SMART Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instructions below can be used to create the SMART Conda
environment.

.. code:: bash

   mkdir SMART_DIR #(pick a name of your choice)
   cd SMART_DIR
   git clone https://gitlab.kitware.com/smart/watch.git
   cd watch
   conda env create -f deployment/conda/conda_env.yml
   conda activate watch

To deactivate the watch environment, run:

.. code:: bash

   conda deactivate

To remove the watch environment, run:

.. code:: bash

   conda deactivate #(if watch is activated before)
   conda remove --name watch --all

To update the watch environment when new packages have been added, run:

.. code:: bash

   conda activate watch
   conda env update -f deployment/conda/conda_env.yml

Installation
~~~~~~~~~~~~

The WATCH Python module can then be installed with ``pip`` via the
following command, where ``/path/to/watch`` is the absolute path to the
directory containing this README.md file.

::

   pip install -e /path/to/watch

Running the Algorithm Toolkit (ATK) example project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that you have already setup the SMART Conda enviornment.

Then enter the following commands in your terminal to run the ATK
example project:

.. code:: bash

   cd atk/example
   alg run

Point your browser to http://localhost:5000/. You should see the
development environment welcome page.

Refer to the `development enviornment`_ portion of the `atk docs`_ for a
crash course on how to use the web-based development environment.

Running tests
-------------

We’re using the ``pytest`` module for running unit tests. Unit

.. _development enviornment: https://algorithm-toolkit.readthedocs.io/en/latest/dev-environment.html#
.. _atk docs: https://algorithm-toolkit.readthedocs.io/en/latest/index.html

.. |master-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/master/latest
.. |master-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg

