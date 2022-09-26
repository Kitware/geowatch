__doc__="
Tries to ensures development version of Jon's libs are installed.
This makes the assumption the repos are already checked out.
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
            pip uninstall "$name" -y
            pip uninstall "$name" -y
            pip install -e "$dpath"
        fi
    else
        echo "does not exist dpath = $dpath"
    fi
done



#for name in "${mylibs[@]}" 
#do
#    echo "name = $name"
#    dpath=$HOME/code/$name
#    if [[ -d $dpath ]]; then
#        gup
#        echo "does not exist dpath = $dpath"
#    fi
#done

for name in "${mylibs[@]}" 
do
    python -c "import $name; print($name.__version__, $name.__file__)"
done
