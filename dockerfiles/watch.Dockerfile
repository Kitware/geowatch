# syntax=docker/dockerfile:1.5.0

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.


ARG BASE_IMAGE=pyenv:311

FROM $BASE_IMAGE

ENV HOME=/root
ENV PIP_ROOT_USER_ACTION=ignore

## Install Prerequisites 
RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        ffmpeg vim tmux jq tree p7zip-full rsync libgsl-dev
apt-get clean 
rm -rf /var/lib/apt/lists/*
EOF

#### UNCOMMENT FOR DEBUGGING
## precache big pip packages if we are debugging steps after this
#RUN <<EOF
##!/bin/bash
#source $HOME/activate
#pip install pip -U
#pip install torch==1.11.0
#EOF


WORKDIR /root
RUN mkdir -p /root/code

# Stage the watch source
COPY setup.py               /root/code/watch/
COPY pyproject.toml         /root/code/watch/
COPY run_developer_setup.sh /root/code/watch/
COPY dev/make_strict_req.sh /root/code/watch/dev/make_strict_req.sh
COPY requirements           /root/code/watch/requirements
COPY watch                  /root/code/watch/watch

#RUN echo $(pwd)

ARG BUILD_STRICT=0

#SHELL ["/bin/bash", "--login", "-c"]

# Setup primary dependencies
# Note: special syntax for caching deps
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
#source $HOME/activate

echo "Preparing to pip install watch"

which python
which pip
python --version
pip --version
pwd
ls -altr
cd /root/code/watch
pwd
ls -altr

echo "Run GEOWATCH developer setup:"
WATCH_STRICT=$BUILD_STRICT WITH_MMCV=1 WITH_DVC=1 WITH_TENSORFLOW=1 WITH_AWS=1 WITH_APT_ENSURE=0 bash run_developer_setup.sh

EOF


#### Copy over the rest of the repo structure
COPY .git          /root/code/watch/.git


# Run simple tests
RUN <<EOF
#!/bin/bash
#source $HOME/activate

echo "Start simple tests"
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m watch --help
EOF

# Copy over the rest of the repo
COPY . /root/code/watch

WORKDIR /root/code/watch

RUN <<EOF
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
echo "

    # SeeAlso:
    # ~/code/watch-smartflow-dags/prepare_system.sh

    # docker login
    # docker pull docker/dockerfile:1.3.0-labs

    #### You need to build the pyenv image first:
    # ./pyenv.Dockerfile

    # An invocation for basic end-to-end building is:

    # Build the pyenv image
    cd $HOME/code/watch
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t pyenv:311 \
        --build-arg PYTHON_VERSION=3.11.2 \
        -f ./dockerfiles/pyenv.Dockerfile .

    # Build the watch image
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "watch:311-strict" \
        --build-arg BUILD_STRICT=1 \
        --build-arg BASE_IMAGE=pyenv:311 \
        -f ./dockerfiles/watch.Dockerfile .

    docker run \
        --volume "$HOME/code/watch":/host-watch:ro \
        --runtime=nvidia -it watch:311-strict bash

   # Will need to bake in a model
   # For futher instructions see: 
   # ../docs/smartflow_running_the_system.rst
"
EOF
