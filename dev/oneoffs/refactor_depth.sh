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

cgrep "from lydorn_utils" 
cgrep "from frame_field_learning"
cgrep "import torch_lydorn"
cgrep "torch_lydorn"


sedr "from lydorn_utils" "from watch.tasks.depth.modules.lydorn_utils"
sedr "from frame_field_learning" "from watch.tasks.depth.modules.frame_field_learning"
sedr "import torch_lydorn" "from watch.tasks.depth.modules.torch_lydorn"
rm watch/tasks/depth/modules/lydorn_utils/setup.py


autoflake ~/code/watch/watch/tasks/depth/modules/torch_lydorn/kornia/augmentation/augmentations.py -i
autoflake ~/code/watch/watch/tasks/depth/modules/torch_lydorn/torchvision/datasets/xview2_dataset.py -i
