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

   conda create -n geowatch python=3.10
   conda activate geowatch

   conda install gdal jq scikit-learn ffmpeg -c conda-forge

   # If you don't have a GPU
   conda install pytorch torchvision cpuonly -c pytorch

   # If you have a NVIDIA GPU (untested)
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch nvidia


.. .. pip install msvc-runtime


Now that these initial requirements have been satisfied we will use pip to
install the rest of the requirements:

.. code:: bash

   pip install geowatch[headless]

   # Or install from source
   # git clone https://gitlab.kitware.com/computer-vision/geowatch.git
   # cd geowatch
   # pip install -e .[headless]


Running The Tutorial
~~~~~~~~~~~~~~~~~~~~

Unfortunately I've been unable to figure out how to get the bash kernel for
Jupyter notebooks to work on Windows. To run the tutorial you will need to run
the bash commands directly in msys or git-bash.

From the start menu, run git-bash. Then we need to activate our conda virtualenv:

.. code:: bash

   conda activate geowatch


Now run the tutorial commands one at a time.



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
