__doc__='
Summary of refactoring made to the watch module on 2021-07-14

To see rough module tree:

sudo apt install tree
tree -I "*.pyc|__pycache__|*.egg-info|htmlcov|logs"
'


sedr(){
    __doc__="""
    Recursive sed

    Args:
        search 
        replace
        pattern (defaults to *.py)

    Example:
        source ~/code/watch/dev/refactor_2021_07_14.sh
        sedr foobefoobe barbebarbe
    """
    SEARCH="${SEARCH:=${1}}"
    REPLACE="${REPLACE:=${2}}"
    PATTERN="${PATTERN:=${3:='*.py'}}"
    LIVE_RUN="${LIVE_RUN:=${4:='False'}}"

    echo "
    === sedr ===
    argv[1] = SEARCH = '$SEARCH' - text to search
    argv[2] = REPLACE = '$REPLACE' - text to replace
    argv[3] = PATTERN = '$PATTERN' - filename patterns to match
    argv[4] = LIVE_RUN = '$LIVE_RUN' - set to 'True' to do the run for real
    "

    find . -type f -iname "${PATTERN}" 

    if [[ "$LIVE_RUN" == "True" ]]; then
        find . -type f -iname "${PATTERN}" -exec sed -i "s|${SEARCH}|${REPLACE}|g" {} + 
    else
        # https://unix.stackexchange.com/questions/97297/how-to-report-sed-in-place-changes
        #find . -type f -iname "${PATTERN}" -exec sed "s|${SEARCH}|${REPLACE}|g" {} + | grep "${REPLACE}"
        find . -type f -iname "${PATTERN}" -exec sed --quiet "s|${SEARCH}|${REPLACE}|gp" {} + | grep "${REPLACE}" -C 100
    fi
}


git mv watch/io/digital_globe.py  watch/gis/digital_globe.py
git mv watch/tools/digital_globe.py  watch/gis/digital_globe.py
git mv watch/scripts watch/cli
git mv watch/tools/hello_world.py watch/cli/hello_world.py
git mv watch/tools/kwcoco_extensions.py watch/utils/kwcoco_extensions.py

git rm -rf watch/tools
rm -rf watch/tools
rm -rf watch/util

sedr 'watch\.tools\.hello_world' 'watch.cli.hello_world' "*.py" True
sedr 'watch\.scripts' 'watch.cli' "*.py" True
sedr 'watch\.tools' 'watch.utils' "*.py" True


mkinit -m watch.utils --lazy --noattr -w
mkinit -m watch.gis --lazy --noattr -w
mkinit -m watch.datasets --lazy --noattr -w
mkinit -m watch.tasks --lazy --noattr -w
