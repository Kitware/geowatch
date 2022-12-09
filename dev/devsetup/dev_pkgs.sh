#!/bin/bash
__doc__="
Tries to ensures development version of Jon's libs are installed.
This makes the assumption the repos are already checked out.

Usage:
source ~/code/watch/dev/devsetup/dev_pkgs.sh
"

mylibs=(
ubelt
mkinit
xdoctest
scriptconfig
cmd_queue
torch_liberator
delayed_image
kwarray
kwimage
kwplot
kwcoco
ndsampler
)


### Pull and update
for name in "${mylibs[@]}" 
do
    echo "name = $name"
    dpath=$HOME/code/$name
    if [[ -d "$dpath" ]]; then
        git fetch
        #(cd "$dpath" && gup)
        (cd "$dpath" && python ~/local/git_tools/git_devbranch.py update)
    else
        echo "does not exist dpath = $dpath"
    fi
done

needs_uninstall=()
needs_install=()
for name in "${mylibs[@]}" 
do
    echo "name = $name"
    dpath=$HOME/code/$name
    if [[ -d $dpath ]]; then
        #base_fpath=$(python -c "import $name; print($name.__file__)")
        if python -c "import sys, $name; sys.exit(1 if 'site-packages' in $name.__file__ else 0)"; then
            echo "already have dpath = $dpath"
        else
            echo "ensuring dpath = $dpath"
            needs_uninstall+=("$name")
            needs_install+=("$dpath")
            #pip uninstall "$name" -y
            #pip uninstall "$name" -y
            #pip install -e "$dpath"
        fi
    else
        echo "does not exist dpath = $dpath"
    fi
done

bash_array_repr "${needs_uninstall[@]}"
bash_array_repr "${needs_install[@]}"

if [[ ${#needs_uninstall[@]} -gt 0 ]]; then
    pip uninstall -y "${needs_uninstall[@]}"
fi

if [[ ${#needs_install[@]} -gt 0 ]]; then
    pip install -e "${needs_install[@]}"
fi


for name in "${mylibs[@]}" 
do
    python -c "import $name; print($name.__version__, $name.__file__)"
done
