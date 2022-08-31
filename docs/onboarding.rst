****************
Watch Onboarding
****************

The purpose of this document is to guide a new user in the installation and
usage of the WATCH system.

This document assumes you are proficient with Python and have an understanding
of virtual environments.

See the README for how to install on to a local machine. This will guide you
through:

* Installing non-Python requirements
* Installing Pyenv and settting up a virtual environment
* Installing the watch package.


Tutorials
=========

See watch/tasks/fusion/experiments/crall/toy_experiments_rgb.sh


The "smartwatch_dvc" command
============================

We provide a utility to help manage data paths called "smartwatch_dvc".  It
comes preconfigured with common paths for core-developer machines You can see
what paths are available by using the "list" command

.. code:: bash

    smartwatch_dvc list

which outputs something like this:


.. code::

                   name hardware         tags                                                               path  exists
    0    drop4_expt_ssd      ssd  phase2_expt                            /root/data/dvc-repos/smart_expt_dvc-ssd   False
    1    drop4_data_ssd      ssd  phase2_data                            /root/data/dvc-repos/smart_data_dvc-ssd   False
    2    drop4_expt_hdd      hdd  phase2_expt                                /root/data/dvc-repos/smart_expt_dvc   False
    3    drop4_data_hdd      hdd  phase2_data                                /root/data/dvc-repos/smart_data_dvc   False


To see full help use `smartwatch_dvc --help`

.. code:: bash

    usage: FindDVCConfig 

    Command line helper to find the path to the watch DVC repo

    positional arguments:
      command               can be find, set, add, list, or remove
      name                  specify a name to query or store or remove

    options:
      -h, --help            show this help message and exit
      --command COMMAND     can be find, set, add, list, or remove (default: find)
      --name NAME           specify a name to query or store or remove (default: None)
      --hardware HARDWARE   Specify hdd, ssd, etc..., Setable and getable property (default: None)
      --priority PRIORITY   Higher is more likely. Setable and getable property (default: None)
      --tags TAGS           User note. Setable and queryable property (default: None)
      --path PATH           The path to the dvc repo. Setable and queryable property (default: None)
      --verbose VERBOSE     verbosity mode (default: 1)
      --must_exist MUST_EXIST
                            if True, filter to only directories that exist. Defaults to false except on "find", which is True. (default: auto)
      --config CONFIG       special scriptconfig option that accepts the path to a on-disk configuration file, and loads that into this 'FindDVCConfig' object. (default: None)
      --dump DUMP           If specified, dump this config to disk. (default: None)
      --dumps               If specified, dump this config stdout (default: False)
