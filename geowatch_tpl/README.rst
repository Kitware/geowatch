GeoWATCH TPL
============

A package that holds third-party libraries that some geowatch tasks require.
This package is installed with geowatch itself. Simply running:

.. code:: python

   import geowatch_tpl


Makes some of the submodules in the repo available. Others need to be explicitly accessed via:

.. code:: python

   import geowatch_tpl
   module = geowatch_tpl.import_submodule('<modname>')


In the future all TPL modules will need to be explicitly requested.

In terms of subfolders we have:

* modules - deprecated, dont add new things here

* submodules - actual git submodules. Only developers should care about these.

* submodules_static - auto-copied versions of git submodules used in production.


Updating Dynamic Submodules
---------------------------

If you want to develop the submodules themselves, then update them dynamically.
Otherwise use the static copies.

.. code:: bash

   git submodule update --init --recursive

Updating Static Submodules
--------------------------

A static "working state" of each submodule is copied into the
geowatch_tpl/static_submodules directory and is what will be used in CI/CD and
production environments.

If you are a developer run the "snapshot_submodules.py" script in this
directory to update the static variant based on your dynamic copy.

.. code:: bash

   cd $HOME/code/watch/geowatch_tpl

   python snapshot_submodules.py
