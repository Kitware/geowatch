Submodules
----------

This directory contains git submodules for TPL repos that are need to be
actively developed. Only developers that are working on these submodules should
need to do anything with them.


A static "working state" of each submodule is copied into the
geowatch_tpl/static_submodules directory and is what will be used in CI/CD and
production environments.

If you are a developer run the "snapshot_submodules.py" script in this
directory to update the static variant based on your dynamic copy.
