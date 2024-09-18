Submodules
----------

This directory contains git submodules for TPL repos that are need to be
actively developed. Only developers that are working on these submodules should
need to do anything with them.


See the `README <../README.rst>`_ in the parent folder for details.


Adding a new submodule
----------------------

Here is what the steps were for jsonargparse



.. code:: bash

    cd "$HOME"/code/geowatch/geowatch_tpl/submodules

    git submodule add git@github.com:Erotemic/jsonargparse.git

    # Update to whatever default branch you want to.

    cd jsonargparse

    git checkout geowatch-fork


Need to register the submodule in ``~/code/geowatch/geowatch_tpl/__init__.py`` along with specific directories that need to be copied.

Also need to register in ``~/code/geowatch/setup.py`` so it is included in the wheel

.. code:: bash

    cd "$HOME"/code/geowatch/geowatch_tpl

    python -m geowatch_tpl.snapshot_submodules
