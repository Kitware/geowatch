WATCH - Wide Area Terrestrial Change Hypercube
==============================================

.. The large version wont work because github strips rst image rescaling. 
.. image:: https://ipfs.io/ipfs/QmYftzG6enTebF2f143KeHiPiJGs66LJf3jT1fNYAiqQvq
   :height: 100px
   :align: left

|master-pipeline| |master-coverage|


This repository addresses the algorithmic challenges of the 
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ (Space-based
Machine Automated Recognition Technique) program.  The goal of this software is
analyze space-based imagery to perform broad-area search for natural and
anthropogenic events and characterize their extent and progression in time and
space.


The following table provides links to relevant resources for the SMART WATCH project:

+------------------------------------+----------------------------------------------------------------+
| The SMART WATCH Python Module      | https://gitlab.kitware.com/watch/watch/                        |
+------------------------------------+----------------------------------------------------------------+

.. .. Under construction
.. .. | The SMART WATCH DVC Repo           | https://gitlab.kitware.com/watch/smart_watch_dvc/              |
.. .. +------------------------------------+----------------------------------------------------------------+

To contribute, please read the `contribution instructions <contribution_instructions.rst>`_.

For information on testing please see `running and writing watch tests <testing_practices.rst>`_.


Getting Started
---------------

To quickly get started locally, clone the watch repo and run:

.. code:: bash

   ./run_developer_setup.py
 

For more details see the `installing watch for development guide <docs/installing_watch.rst.rst>`_.



Internal Development
--------------------

For internal collaberators, please refer to the `internal docs <docs/internal_resources.rst>`_ 

For new internal collaberators, please refer to the `onboarding docs <docs/onboarding.rst>`_ 



.. _development environment: https://algorithm-toolkit.readthedocs.io/en/latest/dev-environment.html#

.. |master-pipeline| image:: https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/smart/watch/-/pipelines/master/latest
.. |master-coverage| image:: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
   :target: https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg
