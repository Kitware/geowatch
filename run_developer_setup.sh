#!/bin/bash
__doc__='
Install watch development environment

CommandLine:
    cd $HOME/code/watch
    ./run_developer_setup.sh
'


if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "NOT INSIDE OF A VIRTUALENV. This script may not run correctly"
fi 

#pip install -r requirements/no-deps.txt

# Install the watch module in development mode
pip install -e .

# Install more fragile dependencies
# pip install imgaug>=0.4.0
# pip install netharn>=0.5.16
pip install GDAL>=3.3.1,!=3.4.0,!=3.3.3 --find-links https://girder.github.io/large_image_wheels -U
pip install GDAL==3.3.1 --find-links https://girder.github.io/large_image_wheels -U

pip install dvc[all]

fix_opencv_conflicts(){
    __doc__="
    Check to see if the wrong opencv is installed, and perform steps to clean
    up the incorrect libraries and install the desired (headless) ones.
    "
    # Fix opencv issues
    pip freeze | grep "opencv-python=="
    HAS_OPENCV_RETCODE="$?"
    pip freeze | grep "opencv-python-headless=="
    HAS_OPENCV_HEADLESS_RETCODE="$?"

    # VAR == 0 means we have it
    if [[ "$HAS_OPENCV_HEADLESS_RETCODE" == "0" ]]; then
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            pip uninstall opencv-python opencv-python-headless -y
            pip install opencv-python-headless
        fi
    else
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            pip uninstall opencv-python -y
        fi
        pip install opencv-python-headless
    fi
}


# Simple tests
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m watch --help
EAGER_IMPORT=1 python -m watch hello_world
