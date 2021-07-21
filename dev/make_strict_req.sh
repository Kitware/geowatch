#!/bin/bash
__doc__="""
Make a strict version of requirements

./dev/req_replacer.sh
"""

#sedr(){
#    __doc__="""
#    Recursive sed
#    """
#    SEARCH=${SEARCH:=${1}}
#    REPLACE=${REPLACE:=${2}}
#    PATTERN=${PATTERN:=${3:='*.py'}}
#    LIVE_RUN=${LIVE_RUN:=${3:='False'}}

#    find . -type f -iname "${PATTERN}" 

#    if [[ "$LIVE_RUN" == "True" ]]; then
#        find . -type f -iname "${PATTERN}" -exec sed -i "s|${SEARCH}|${REPLACE}|g" {} + 
#    else
#        # https://unix.stackexchange.com/questions/97297/how-to-report-sed-in-place-changes
#        #find . -type f -iname "${PATTERN}" -exec sed "s|${SEARCH}|${REPLACE}|g" {} + | grep "${REPLACE}"
#        find . -type f -iname "${PATTERN}" -exec sed --quiet "s|${SEARCH}|${REPLACE}|gp" {} + | grep "${REPLACE}" -C 100
#    fi
#}
#PATTERN=conda_env.yml

# Make strict version of requirements

sed 's/requirements/requirements-strict/g' conda_env.yml > conda_env_strict.yml
sed -i 's/>=/==/g' conda_env_strict.yml 

mkdir -p requirements-strict
sed 's/>=/==/g' requirements/production.txt > requirements-strict/production.txt
sed 's/>=/==/g' requirements/development.txt > requirements-strict/development.txt
sed 's/>=/==/g' requirements/optional.txt > requirements-strict/optional.txt
sed 's/>=/==/g' requirements/problematic.txt > requirements-strict/problematic.txt
