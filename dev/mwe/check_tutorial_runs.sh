# Run a docker image that forwards ports to the host system
docker run --ipc=host --volume "$HOME"/.cache/pip:/pip_cache -it -p 8888:8888 python:3.11 bash


###############
# INSIDE DOCKER
###############

# Ensure ffmpeg is installed
apt update -y
apt install ffmpeg -y

# Install the geowatch package with extras
export PIP_CACHE_DIR=/pip_cache
pip install geowatch[headless,development,optional]

# Finalize special dependencies
# python -m watch.cli.special.finish_install
pip install --prefer-binary GDAL>=3.4.1 --find-links https://girder.github.io/large_image_wheels


# Sanity check
EAGER_IMPORT=1 python -m watch --help


## Shell version
#curl -LJO https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.sh
#source tutorial1_rgb_network.sh


# Install jupyter notebook and the bash kernel for the tutorial
pip install jupyter bash_kernel
python -m bash_kernel.install

# Jupyter Version
curl -LJO https://gitlab.kitware.com/computer-vision/geowatch/-/raw/main/tutorial/tutorial1_rgb_network.ipynb
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root  tutorial1_rgb_network.ipynb


## On the host system, navigate to localhost:8888 which will bring up the
#notebook. Then follow the steps in the notebook.
