#!/bin/bash
__doc__="
Make a strict version of requirements

./dev/make_strict_req.sh
"
# Make strict version of requirements
#sed 's/requirements/requirements-strict/g' conda_env.yml > conda_env_strict.yml
#sed -i 's/>=/==/g' conda_env_strict.yml

mkdir -p geowatch/rc/requirements-strict
for fpath in geowatch/rc/requirements/*.txt; do
    echo "Making strict version of fpath = $fpath"
    fname=$(python -c "import pathlib; print(pathlib.Path('$fpath').name)")
    sed 's/>=/==/g' "requirements/$fname" > "requirements-strict/$fname"
done
