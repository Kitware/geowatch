#!/bin/bash
__doc__="
Make a strict version of requirements

./dev/make_strict_req.sh
"
# Make strict version of requirements
#sed 's/requirements/requirements-strict/g' conda_env.yml > conda_env_strict.yml
#sed -i 's/>=/==/g' conda_env_strict.yml
#
#
LOOSE_REQUIREMENTS_DPATH=geowatch/rc/requirements
STRICT_REQUIREMENTS_DPATH=geowatch/rc/requirements-strict

mkdir -p "$STRICT_REQUIREMENTS_DPATH"
for fpath in "$LOOSE_REQUIREMENTS_DPATH"/*.txt; do
    echo "Making strict version of fpath = $fpath"
    fname=$(python -c "import pathlib; print(pathlib.Path('$fpath').name)")
    sed 's/>=/==/g' "$LOOSE_REQUIREMENTS_DPATH/$fname" > "$STRICT_REQUIREMENTS_DPATH/$fname"
done
