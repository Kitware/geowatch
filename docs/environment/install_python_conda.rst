To install Miniconda3, follow the instructions below for Linux. For Windows 11
users, the Windows Subsystem for Linux (WSL) allows you to run Linux within
Windows.

.. code:: bash

    # Download the conda install script into a temporary directory
    mkdir -p ~/tmp
    cd ~/tmp

    # To update to a newer version see:
    # https://docs.conda.io/en/latest/miniconda_hashes.html for updating
    CONDA_INSTALL_SCRIPT=Miniconda3-py39_4.12.0-Linux-x86_64.sh
    curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT

    # For security, it is important to verify the hash
    CONDA_EXPECTED_SHA256=1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
    echo "${CONDA_EXPECTED_SHA256}  ${CONDA_INSTALL_SCRIPT}" > conda_expected_hash.sha256
    if ! sha256sum --status -c conda_expected_hash.sha256; then
        echo "Downloaded file does not match hash! DO NOT CONTINUE!"
    else
        echo "Hash verified, continue with install"
        chmod +x $CONDA_INSTALL_SCRIPT
        # Install miniconda to user local directory
        _CONDA_ROOT=$HOME/.local/conda
        sh $CONDA_INSTALL_SCRIPT -b -p $_CONDA_ROOT
        # Activate the basic conda environment
        source $_CONDA_ROOT/etc/profile.d/conda.sh
        # Update the base
        conda update --name base conda --yes
    fi

NOTE: If using conda, do NOT use ``conda install`` to install Python packages.


Create GEOWATCH Environment with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using conda, the instructions below can be used to create the GEOWATCH Conda
environment.

.. code:: bash

   mkdir WATCH_DIR #(pick a name of your choice)
   cd WATCH_DIR
   git clone https://gitlab.kitware.com/computer-vision/geowatch.git
   cd geowatch
   conda env create -f conda_env.yml
   conda activate geowatch

To deactivate the geowatch environment, run:

.. code:: bash

   conda deactivate

To remove the geowatch environment, run:

.. code:: bash

   conda deactivate #(if geowatch is activated before)
   conda remove --name geowatch --all

To update the geowatch environment when new packages have been added, run:

.. code:: bash

   conda activate geowatch
   conda env update -f deployment/conda/conda_env.yml
