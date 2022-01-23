#!/bin/bash
__doc__="""
Make a strict version of requirements

./dev/make_strict_req.sh
"""
# Make strict version of requirements
sed 's/requirements/requirements-strict/g' conda_env.yml > conda_env_strict.yml
sed -i 's/>=/==/g' conda_env_strict.yml 

mkdir -p requirements-strict
sed 's/>=/==/g' requirements/production.txt > requirements-strict/production.txt
sed 's/>=/==/g' requirements/development.txt > requirements-strict/development.txt
sed 's/>=/==/g' requirements/optional.txt > requirements-strict/optional.txt
sed 's/>=/==/g' requirements/problematic.txt > requirements-strict/problematic.txt
sed 's/>=/==/g' requirements/dvc_and_aws.txt > requirements-strict/dvc_and_aws.txt
