GEOWATCH On Windows
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

   conda create -n geowatch2 python=3.10 -y
   conda activate geowatch2

   conda install gdal jq scikit-learn ffmpeg -c conda-forge -y

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

    conda install curl
    curl https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.sh -O tutorial1_rgb_network.sh

    # Show the tutorial (it's readable)
    cat tutorial1_rgb_network.sh

    # The tutorial is self-executing.
    ./tutorial1_rgb_network.sh


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

Which means you need to enable virtualization in your BIOS.

And I haven't gotten farther than that, but if you have it setup, it is much
easier to run the software on Ubuntu.
