#!/bin/bash
__doc__="
Tries to ensures development version of Jon's libs are installed.
This makes the assumption the repos are already checked out.

Requirements:
    # For auto-branch upgrades
    pip install git_well

    # Not the best way, but a way.
    curl https://raw.githubusercontent.com/Erotemic/local/main/init/utils.sh > erotemic_utils.sh

Usage:
    source ~/code/geowatch/dev/devsetup/dev_pkgs.sh
"

# Place where the source packages are located
CODE_DPATH=$HOME/code

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
kwutil
ndsampler
simple_dvc
)


DO_FETCH=1
DRY_RUN=0


if [[ "$DO_FETCH" == "1" ]]; then
    ### Pull and update
    for name in "${mylibs[@]}"
    do
        echo "name = $name"
        dpath=$CODE_DPATH/$name
        if [ -d "$dpath" ]; then
            #git fetch
            #(cd "$dpath" && gup)
            echo "dpath = $dpath"
            #(cd "$dpath" && git fetch && python ~/local/git_tools/git_devbranch.py update)
            (cd "$dpath" && git fetch && git-well branch_upgrade)
        else
            echo "does not exist dpath = $dpath"
        fi
    done
fi

echo "
My Libs:"
bash_array_repr "${mylibs[@]}"

needs_uninstall=()
needs_install=()
for name in "${mylibs[@]}"
do
    echo "Check: name = $name"
    dpath=$CODE_DPATH/$name
    if [[ -d $dpath ]]; then
        #base_fpath=$(python -c "import $name; print($name.__file__)")
        if python -c "import sys, $name; sys.exit(1 if 'site-packages' in $name.__file__ else 0)"; then
            echo " * already have dpath = $dpath"
        else
            echo " * will ensure dpath = $dpath"
            needs_uninstall+=("$name")
            needs_install+=("-e" "$dpath")
            #pip uninstall "$name" -y
            #pip uninstall "$name" -y
            #pip install -e "$dpath"
        fi
    else
        echo " * does not exist dpath = $dpath"
    fi
done

echo "
Needs Uninstall:"
bash_array_repr "${needs_uninstall[@]}"


echo "
Needs Install:"
bash_array_repr "${needs_install[@]}"



if [[ "$DRY_RUN" == "0" ]]; then

    echo "
    Uninstalling:
    "
    if [[ ${#needs_uninstall[@]} -gt 0 ]]; then
        pip uninstall -y "${needs_uninstall[@]}"
    fi

    echo "
    Finished Uninstalling.

    Installing:
    "
    if [[ ${#needs_install[@]} -gt 0 ]]; then
        # * Disable build isolation because it is faster and we usually wont need it.
        # * Note the -e needs to be before every package, this is handled earlier
        pip install --no-build-isolation "${needs_install[@]}"
    fi

    echo "
    Finished Installing
    "
fi


echo "
Check that the installed versions / paths are what you expect:
"
for name in "${mylibs[@]}"
do
    python -c "import $name; print(f'{$name.__name__:<17} - {$name.__version__} - {$name.__file__}')"
done
