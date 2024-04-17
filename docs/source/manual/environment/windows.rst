GeoWATCH On Windows
===================

Windows usage is far less tested than Linux. If possible use Linux, but we will
try to support windows.  This document outlines high level steps to get
geowatch running on windows.  We will use miniconda3 for Python. Other options
will works.

Steps I used on windows 10:


Installer Miniconda
~~~~~~~~~~~~~~~~~~~

Download `Miniconda3 Windows 64-bit installer <https://docs.conda.io/en/latest/miniconda.html>`_ (with Python 3.10 as the default).

Run the installer (using default settings).


Install Bash
~~~~~~~~~~~~

We will also need msys or git-bash

Download `the git-bash 64-bit standalone installer <https://git-scm.com/download/win>` and install it with default settings.


Environment Setup
~~~~~~~~~~~~~~~~~

From the start menu, run the Anaconda Prompt (miniconda).

BEFORE installing geowatch we will need to use conda to get some dependencies
that don't have windows binaries hosted on pypi.

It is recommened to be in a virtual enviornment

.. code:: bash

   conda create -n geowatch python=3.10 -y
   conda activate geowatch

   conda install gdal jq scikit-learn ffmpeg curl -c conda-forge -y

   # If you don't have a GPU
   conda install pytorch torchvision cpuonly -c pytorch -y

   # If you have a NVIDIA GPU (untested)
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch nvidia -y


.. .. pip install msvc-runtime


Now that these initial requirements have been satisfied we will use pip to
install the rest of the requirements:

.. code:: bash

   pip install geowatch[headless]


Optional: Installing From Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the source code directly, open an anaconda prompt and run:

.. code:: bash

    # Or the path to wherever you installed the git bash executable
    "C:\Program Files\Git\bin\bash.exe"

    source $USERPROFILE/miniconda3/etc/profile.d/conda.sh
    conda activate geowatch

    git clone https://gitlab.kitware.com/computer-vision/geowatch
    cd geowatch
    pip install -e .[headless]

Running The Tutorial
~~~~~~~~~~~~~~~~~~~~

Unfortunately I've been unable to figure out how to get the bash kernel for
Jupyter notebooks to work on Windows. To run the tutorial you will need to run
the bash commands directly in msys or git-bash.

From the start menu, open an anaconda prompt. In this shell we need to start bash and then activate our conda virtualenv:


.. code:: bash

    "C:\Program Files\Git\bin\bash.exe"

    source $USERPROFILE/miniconda3/etc/profile.d/conda.sh
    conda activate geowatch


The bash tutorial lives `here
<https://gitlab.kitware.com/computer-vision/geowatch/-/blob/main/tutorial/tutorial1_rgb_network.sh>`_,
and can be downloaded and run via:

.. code:: bash

    curl -LJO https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.sh

    # Show the tutorial (it's readable)
    cat tutorial1_rgb_network.sh

    # The tutorial is self-executing.
    source tutorial1_rgb_network.sh


OR if you cloned the source repo:

.. code:: bash

    cd geowatch
    ./tutorial/tutorial1_rgb_network.sh


Or you can run the tutorial commands one at a time by copy / pasting commands
from the script into your terminal.



WSL2 Instructions
-----------------

Ensure you have WSL2 enabled:

In the start menu search for "Turn Windows Features On or Off". A Windows
Features dialog will pop up. Scroll down to "Windows Subsystem for Linux" and
ensure it is checked. Press OK, and restart your computer when prompted.

After rebooting type "powershell" in the start menu. Right click the powershell
icon and click run as administrator.

.. code:: bash


    wsl --install -d Ubuntu


You might be prompted to visit a web page for more information. You will need
to do this to download and install a WSL2 Linux kernel update package.


You might get an error:

"Error: 0x80370102 The virtual machine could not be started because a required feature is not installed."

Which means you need to enable virtualization in your BIOS, AND ensure that
"Virtual Machine Platform" windows feature is enabled.

https://support.microsoft.com/en-us/windows/enable-virtualization-on-windows-11-pcs-c5578302-6e43-4b4b-a449-8ced115f58e1


If this works you will be prompted to enter a new username/password.


Running on WSL2
~~~~~~~~~~~~~~~

If you have a working WSL2 prompt, install conda:


.. code:: bash

    # Download the conda install script into a temporary directory
    mkdir -p ~/tmp
    cd ~/tmp

    # To update to a newer version see:
    # https://docs.conda.io/en/latest/miniconda_hashes.html for updating
    CONDA_INSTALL_SCRIPT=Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
    curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT

    # For security, it is important to verify the hash
    CONDA_EXPECTED_SHA256=aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651
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


Create an activate the virtual env

.. code:: bash

   conda create -n geowatch python=3.10 -y
   conda activate geowatch

Install geowatch and GDAL:

.. code:: bash

    pip install geowatch[headless,development,optional]

    pip install --prefer-binary GDAL>=3.4.1 --find-links https://girder.github.io/large_image_wheels

You will also want to install ffmpeg:

.. code:: bash

    sudo apt update -y
    sudo apt install ffmpeg -y


It also may be necessary to install the following packages:

.. code:: bash

    sudo apt install libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-render-util0, libxcb-xkb1, libxkbcommon-x11-0 -y


You can download the shell version of the tutorial:

.. code:: bash

    curl -LJO https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.sh

    # Can be run directly
    source tutorial1_rgb_network.sh

Or the Jupyter version of the tutorial:

.. code:: bash

    curl -LJO https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.ipynb

    # ensure you have jupyter with bash_kernel installed
    pip install jupyter bash_kernel
    python -m bash_kernel.install

    # Start the notebook
    jupyter notebook tutorial1_rgb_network.ipynb

For Jupyter on WSL you will need to start a browser on your host machine. Use
the URL with the authentication token the ``jupyter notebook`` command printed
out (you should be able to ctrl+click it)
