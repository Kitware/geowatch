# syntax=docker/dockerfile:1.5.0

# **************************************************
# The pyenv dockerfile builds a cuda compatible pyenv
# environment with a specific version precompiled and builtin.
#
# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.
# (docker devs: todo unconsequential heredocs)
#
# **************************************************

#### STAGE 1
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 as pyenv_stage

ARG PYTHON_VERSION=3.11.2
ARG PYENV_VERSION=v2.3.13

ENV HOME=/root
ENV PYENV_ROOT=/root/.pyenv

## Install Prerequisites 
RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget curl \
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl 
apt-get clean 
rm -rf /var/lib/apt/lists/*
EOF

## Install pyenv
RUN <<EOF
#!/bin/bash
git clone https://github.com/pyenv/pyenv.git -b $PYENV_VERSION $PYENV_ROOT 
(cd $PYENV_ROOT && src/configure && make -C src)
EOF


## Use pyenv to compile an optimized Python
RUN <<EOF
#!/bin/bash
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$($PYENV_ROOT/bin/pyenv init -)"

PROFILE_TASK="-m test.regrtest --pgo test_array test_base64 test_binascii test_binhex test_binop test_c_locale_coercion test_csv test_json test_hashlib test_unicode test_codecs test_traceback test_decimal test_math test_compile test_threading test_time test_fstring test_re test_float test_class test_cmath test_complex test_iter test_struct test_slice test_set test_dict test_long test_bytes test_memoryview test_io test_pickle"

PYTHON_CONFIGURE_OPTS="--enable-shared --enable-optimizations --with-computed-gotos --with-lto"

PYTHON_CFLAGS="-march=native -O2 -pipe" 

PROFILE_TASK=$PROFILE_TASK \
PYTHON_CFLAGS="${PYTHON_CFLAGS}" \
PYTHON_CONFIGURE_OPTS="${PYTHON_CONFIGURE_OPTS}" \
pyenv install $PYTHON_VERSION 

EOF

#SHELL ["/bin/bash", "--login", "-c"]

ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# pyenv prefix is not working. We should be able to hack it?
env PYENV_PREFIX=/root/.pyenv/versions/$PYTHON_VERSION
env PYTHON_VERSION=$PYTHON_VERSION


## Setup a default Python virtualenv
## (this does not seem to be reliable)
RUN <<EOF
#!/bin/bash
echo "Init pyenv"
echo "HOME=$HOME"
echo "PYENV_ROOT=$PYENV_ROOT"
echo "PYTHON_VERSION=$PYTHON_VERSION"
echo "PYENV_VERSION=$PYENV_VERSION"
echo "PYENV_PREFIX=$PYENV_PREFIX"
echo "PATH=$PATH"

## Setup global pyenv version
eval "$($PYENV_ROOT/bin/pyenv init -)"
pyenv global $PYTHON_VERSION
pyenv global

# Not sure why I need the unset here
unset PYENV_VERSION

echo "Write bashrc and profile"
BASHRC_CONTENTS='
# Add the pyenv command to our environment if it exists
export HOME="/root"
export PYENV_ROOT="$HOME/.pyenv"
if [ -d "$PYENV_ROOT" ]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$($PYENV_ROOT/bin/pyenv init -)"
    source $PYENV_ROOT/completions/pyenv.bash
fi
'
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
EOF




#### STAGE 2
FROM pyenv_stage as module_stage

ENV HOME=/root

## Install Prerequisites 
RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        ffmpeg vim tmux jq tree p7zip-full rsync
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
COPY setup.py       /root/code/watch/
COPY pyproject.toml /root/code/watch/
COPY requirements   /root/code/watch/requirements
COPY watch          /root/code/watch/watch

#RUN echo $(pwd)

ARG BUILD_STRICT=0


#SHELL ["/bin/bash", "--login", "-c"]

# Setup primary dependencies
# Note: special syntax for caching deps
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
#source $HOME/activate

# Not sure why I need the unset here
unset PYENV_VERSION

echo "
Preparing to pip install watch
"

which python
which pip
python --version
pip --version
pwd
ls -altr

echo "
Pip install latest Python build tools:
"
python -m pip install --prefer-binary pip setuptools wheel build -U

echo "
Pip install watch itself
"
if [ "$BUILD_STRICT" -eq 1 ]; then
    echo "BUILDING STRICT VARIANT"
    pip install --prefer-binary -e /root/code/watch[runtime-strict,development-strict,optional-strict,headless-strict]
else
    echo "BUILDING LOOSE VARIANT"
    pip install --prefer-binary -e /root/code/watch[development,optional,headless]
    # python -m pip install dvc[all]>=2.13.0
    # pip install awscli
fi

EOF


# Finalize more fickle dependencies
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
#source $HOME/activate

# Not sure why I need the unset here
unset PYENV_VERSION

cd /root/code/watch
if [ "$BUILD_STRICT" -eq 1 ]; then
    echo "FINALIZE STRICT VARIANT DEPS"
    sed 's/>=/==/g' requirements/gdal.txt > requirements/gdal-strict.txt
    pip install -r requirements/gdal-strict.txt
else
    echo "FINALIZE LOOSE VARIANT DEPS"
    pip install -r requirements/gdal.txt
fi

EOF


#### Copy over the rest of the repo structure
COPY .git          /root/code/watch/.git


# Install other useful tools
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
#source $HOME/activate

# Not sure why I need the unset here
unset PYENV_VERSION

# python -m pip install dvc[all]>=2.13.0
# pip install scikit-image>=0.18.1
pip install awscli

EOF


# Does this get rid of the unset issue?
ARG PYENV_VERSION=


# Run simple tests
RUN <<EOF
#!/bin/bash
#source $HOME/activate

echo "Start simple tests"
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m geowatch --help
EOF

# Copy over the rest of the repo
COPY . /root/code/watch

WORKDIR /root/code/watch





################
### __DOCS__ ###
################
RUN <<EOF
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
echo "
    # docker login
    # docker pull docker/dockerfile:1.3.0-labs

    # Set this to the path of your watch repo
    export LOCAL_REPO_DPATH=$HOME/code/watch

    # Set this to a place where you can write temp files
    export STAGING_DPATH=$HOME/tmp/watch-img-staging

    cd 
    rm -rf $STAGING_DPATH
    mkdir -p $STAGING_DPATH

    git clone --origin=host-$HOSTNAME $LOCAL_REPO_DPATH/.git $STAGING_DPATH/watch

    # Add the real origin to the repo as a convinience
    cd $STAGING_DPATH/watch
    git remote add origin git@gitlab.kitware.com:smart/watch.git


    #### 3.11 (strict)
    cd $STAGING_DPATH/watch

    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "watch:311-strict" \
        --build-arg BUILD_STRICT=1 \
        --build-arg PYTHON_VERSION=3.11.2 \
        --target module_stage .

    # Tests
    docker run \
        --volume "$HOME/code/watch":/host-watch:ro \
        --runtime=nvidia -it watch:311-strict bash

    git remote add dockerhost /host-watch/.git

    # Optional: Push the container to smartgitlab
    IMAGE_NAME=watch:311-strict
    docker tag $IMAGE_NAME registry.smartgitlab.com/kitware/$IMAGE_NAME
    docker push registry.smartgitlab.com/kitware/$IMAGE_NAME

   # Will need to bake in a model
   # For futher instructions see: 
   # ../docs/smartflow_running_the_system.rst
"
EOF
