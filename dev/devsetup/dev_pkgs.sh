#!/bin/bash
__doc__="
Tries to ensures development version of Jon's libs are installed.
This makes the assumption the repos are already checked out.


Make sure you are already in the venv you want and you have pip installed
git-well and uv.

Requirements:
    # For auto-branch upgrades
    pip install git_well
    pip install uv

Usage:
    bash ~/code/geowatch/dev/devsetup/dev_pkgs.sh
"
set -euo pipefail


check_command_exists() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: Required command '$cmd' not found. Please install it first." >&2
        return 1
    fi
}


bash_array_repr(){
    __doc__='
    Given a bash array, this should print a literal copy-pastable
    representation

    Example:
        ARR=(1 "2 3" 4)
        bash_array_repr "${ARR[@]}"

    Dependency Free Alternative:
        echo "${ARR[@]}"
    ...
    '
    local ARGS=("$@")
    if [ "${#ARGS[@]}" -gt 0 ]; then
        # Not sure if the double or single quotes is better here
        echo "($(printf "'%s' " "${ARGS[@]}"))"
    else
        echo "()"
    fi
}

# Place where the source packages are located
CODE_DPATH=$HOME/code

# Options (currently hard coded)
DO_FETCH=1
DO_INSTALL=1
DO_CLONE=1

mylibs=(
ubelt
mkinit
xdev
xdoctest
scriptconfig
cmd_queue
torch_liberator
delayed_image
kwarray
kwimage
kwplot
kwcoco
kwcoco_dataloader
kwutil
kwgis
ndsampler
simple_dvc
)

declare -A REPO_REMOTE_LUT=(
    ["ubelt"]="https://github.com/Erotemic/ubelt.git"
    ["mkinit"]="https://github.com/Erotemic/mkinit.git"
    ["xdev"]="https://github.com/Erotemic/xdev.git"
    ["xdoctest"]="https://github.com/Erotemic/xdoctest.git"
    ["scriptconfig"]="https://gitlab.kitware.com/utils/scriptconfig.git"
    ["cmd_queue"]="https://gitlab.kitware.com/computer-vision/cmd_queue.git"
    ["torch_liberator"]="https://gitlab.kitware.com/computer-vision/torch_liberator.git"
    ["delayed_image"]="https://gitlab.kitware.com/computer-vision/delayed_image.git"
    ["kwarray"]="https://gitlab.kitware.com/computer-vision/kwarray.git"
    ["kwimage"]="https://gitlab.kitware.com/computer-vision/kwimage.git"
    ["kwplot"]="https://gitlab.kitware.com/computer-vision/kwplot.git"
    ["kwcoco"]="https://gitlab.kitware.com/computer-vision/kwcoco.git"
    ["kwcoco_dataloader"]="https://gitlab.kitware.com/computer-vision/kwcoco_dataloader.git"
    ["kwutil"]="https://gitlab.kitware.com/computer-vision/kwutil.git"
    ["kwgis"]="https://gitlab.kitware.com/computer-vision/kwgis.git"
    ["ndsampler"]="https://gitlab.kitware.com/computer-vision/ndsampler.git"
    ["simple_dvc"]="https://gitlab.kitware.com/computer-vision/simple_dvc.git"
)

if ! check_command_exists uv; then
    echo "Could not find uv, falling back to pip. Install uv for faster installs"
    _PIP_PREFIX=
else
    _PIP_PREFIX=uv
    echo "Found uv"
fi



if [[ "$DO_FETCH" == "1" ]]; then
    echo "====================="
    echo "Start Pull and Update"
    echo "====================="
    check_command_exists git-well || return 1

    ### Pull and update
    for name in "${mylibs[@]}"
    do
        echo "name = $name"
        REPO_DPATH=$CODE_DPATH/$name
        if [ -d "$REPO_DPATH" ]; then
            #git fetch
            #(cd "$REPO_DPATH" && gup)
            echo "REPO_DPATH = $REPO_DPATH"
            #(cd "$REPO_DPATH" && git fetch && python ~/local/git_tools/git_devbranch.py update)
            (cd "$REPO_DPATH" && git fetch && git-well branch_upgrade)
        else
            echo "does not exist REPO_DPATH = $REPO_DPATH"
            if [[ "$DO_CLONE" == "1" ]]; then
                # Hack
                REPO_URL="${REPO_REMOTE_LUT[$name]}"
                REPO_DPATH=$CODE_DPATH/$name
                echo "REPO_URL = $REPO_URL"
                git clone "$REPO_URL" "$REPO_DPATH"
                (cd "$REPO_DPATH" && git fetch && git-well branch_upgrade)
            fi
        fi
    done
else
    echo "Skip Fetching"
fi

echo "
My Libs:"
bash_array_repr "${mylibs[@]}"


echo "====================="
echo "Check for tasks to do"
echo "====================="

needs_uninstall=()
needs_install=()
for name in "${mylibs[@]}"
do
    echo "Check: name = $name"
    REPO_DPATH=$CODE_DPATH/$name
    if [[ -d $REPO_DPATH ]]; then
        #base_fpath=$(python -c "import $name; print($name.__file__)")
        if python -c "import sys, $name; sys.exit(1 if 'site-packages' in $name.__file__ else 0)" 2> /dev/null; then
            echo " * already have REPO_DPATH = $REPO_DPATH"
        else
            echo " * will ensure REPO_DPATH = $REPO_DPATH"
            needs_uninstall+=("$name")
            needs_install+=("-e" "$REPO_DPATH")
            #pip uninstall "$name" -y
            #pip uninstall "$name" -y
            #pip install -e "$REPO_DPATH"
        fi
    else
        echo " * does not exist REPO_DPATH = $REPO_DPATH"
    fi
done

echo "
Needs Uninstall:"
bash_array_repr "${needs_uninstall[@]}"


echo "
Needs Install:"
bash_array_repr "${needs_install[@]}"



if [[ "$DO_INSTALL" == "1" ]]; then
    echo "===================="
    echo "Do Developer Install"
    echo "===================="


    echo "
    Uninstalling:
    "
    if [[ ${#needs_uninstall[@]} -gt 0 ]]; then
        $_PIP_PREFIX pip uninstall -y "${needs_uninstall[@]}"
    fi

    echo "
    Finished Uninstalling.

    Installing:
    "
    if [[ ${#needs_install[@]} -gt 0 ]]; then
        # * Disable build isolation because it is faster and we usually wont need it.
        # * Note the -e needs to be before every package, this is handled earlier
        #echo pip install --no-build-isolation "${needs_install[@]}"
        #pip install --no-build-isolation "${needs_install[@]}"
        # Looks like build isolation is probably important
        echo pip install "${needs_install[@]}"
        # TODO: use uv
        $_PIP_PREFIX pip install "${needs_install[@]}"
    fi

    echo "
    Finished Installing
    "
else
    echo "======================"
    echo "Skip Developer Install"
    echo "======================"
fi


echo "===================="
echo "Check Installed Libs"
echo "===================="
echo "
Check that the installed versions / paths are what you expect:
"
for name in "${mylibs[@]}"
do
    python -c "import $name; print(f'{$name.__name__:<17} - {$name.__version__} - {$name.__file__}')"
done
