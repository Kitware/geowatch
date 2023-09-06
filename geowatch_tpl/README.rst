GEOWATCH TPL
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
