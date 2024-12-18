Submodules
----------

This directory contains git submodules for TPL repos that are need to be
actively developed. Only developers that are working on these submodules should
need to do anything with them.


See the `README <../README.rst>`_ in the parent folder for details.


Adding a new submodule
----------------------


To add a new submodule, you need to know the following information:

.. code:: bash

   # Path to the GeoWATCH repo
   GEOWATCH_REPO_DPATH="$HOME"/code/geowatch
   # Git URL of the submodule to clone
   NEW_SUBMODULE_REMOTE_URL=git@github.com:username/repo.git
   # Name of the branch to use
   NEW_SUBMODULE_BRANCH=main



Examples
^^^^^^^^

Examples of previously used information are as follows for jsonargparse:

.. code:: bash

   # Path to the GeoWATCH repo
   GEOWATCH_REPO_DPATH="$HOME"/code/geowatch
   # Git URL of the submodule to clone
   NEW_SUBMODULE_REMOTE_URL=git@github.com:Erotemic/jsonargparse.git
   NEW_SUBMODULE_NAME=$(git-well url "$NEW_SUBMODULE_REMOTE_URL" repo_name)
   # Name of the branch to use
   NEW_SUBMODULE_BRANCH=geowatch-fork


Examples of previously used information are as follows for detectron2:

.. code:: bash

   # Path to the GeoWATCH repo
   GEOWATCH_REPO_DPATH="$HOME"/code/geowatch
   # Git URL of the submodule to clone
   NEW_SUBMODULE_REMOTE_URL=git@github.com:Erotemic/detectron2.git
   NEW_SUBMODULE_NAME=$(git-well url "$NEW_SUBMODULE_REMOTE_URL" repo_name)
   # Name of the branch to use
   NEW_SUBMODULE_BRANCH=exif_options


Add Procedure
^^^^^^^^^^^^^

Given correct information information, run the following:

.. code:: bash

    # Check that above information was populated correctly
    echo "
    GEOWATCH_REPO_DPATH = $GEOWATCH_REPO_DPATH
    NEW_SUBMODULE_REMOTE_URL = $NEW_SUBMODULE_REMOTE_URL
    NEW_SUBMODULE_NAME = $NEW_SUBMODULE_NAME
    NEW_SUBMODULE_BRANCH = $NEW_SUBMODULE_BRANCH
    "

    # CD into the submodules directory and add the repo as a submodule.
    cd "$GEOWATCH_REPO_DPATH"/geowatch_tpl/submodules
    git submodule add "$NEW_SUBMODULE_REMOTE_URL"
    # CD into the new submodule
    cd "$NEW_SUBMODULE_NAME"
    # Update to whatever default branch you want to.
    git checkout "$NEW_SUBMODULE_BRANCH"


Then there are two manual steps that need to be taken:

* Need to register the submodule in ``~/code/geowatch/geowatch_tpl/__init__.py`` along with specific directories that need to be copied.

* Also need to register in ``~/code/geowatch/setup.py`` so it is included in the wheel

Finally, we can call a helper script to update our submodule snapshots.

.. code:: bash

    cd "$GEOWATCH_REPO_DPATH"/geowatch_tpl

    python -m geowatch_tpl.snapshot_submodules
