Using GeoWATCH DVC
------------------

In order to make reproducing results as easy as copy/pasting commands into a
terminal, we provide the ``geowatch_dvc`` tool to register the paths to their
DVC repos as follows:

When you register your drop4 data / experiment paths, the DVC examples in this
repo will generally work out of the box. The important part is that your path
agrees with the tags used in the examples. Telling the registry if the path
lives on an HDD or SSD is also useful.


.. code:: bash

   # Register the path you cloned the smart_data_dvc and smart_expt_dvc repositories to.
   geowatch_dvc add my_drop4_data --path=$HOME/Projects/SMART/smart_data_dvc --hardware=hdd --priority=100 --tags=phase2_data
   geowatch_dvc add my_drop4_expt --path=$HOME/Projects/SMART/smart_expt_dvc --hardware=hdd --priority=100 --tags=phase2_expt


The examples in this repo will generally use this pattern to query for the
machine-specific data location. Ensure that these commands work and output
the correct paths

.. code:: bash

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    # Test to make sure these work.
    echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
    echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
