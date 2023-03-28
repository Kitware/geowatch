# syntax=docker/dockerfile:1.5.0

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.


ARG BASE_IMAGE=pyenv:311

FROM $BASE_IMAGE

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

# python -m pip install dvc[all]>=2.13.0
# pip install scikit-image>=0.18.1
pip install awscli

EOF


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

# docker login
# docker pull docker/dockerfile:1.3.0-labs


#### You need to build the pyenv image first:
# ./pyenv.Dockerfile

cd $HOME/code/watch

mkdir -p $HOME/tmp/watch-img-staging
git clone --origin=host-$HOSTNAME $HOME/code/watch/.git $HOME/tmp/watch-img-staging/watch
cd $HOME/tmp/watch-img-staging/watch
git remote add origin git@gitlab.kitware.com:smart/watch.git

# Either build the pyenv image or
#docker pull gitlab.kitware.com:4567/smart/watch/pyenv:311
#docker tag gitlab.kitware.com:4567/smart/watch/pyenv:311 pyenv:311 


#### 3.10

cd $HOME/tmp/watch-img-staging/watch
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "watch:310-strict" \
    --build-arg BASE_IMAGE=pyenv:310 \
    --build-arg BUILD_STRICT=1 \
    -f ./dockerfiles/watch.Dockerfile .

docker run \
    --volume "$HOME/code/watch":/host-watch:ro \
    --runtime=nvidia -it watch:310-strict bash

#### 3.11

cd $HOME/tmp/watch-img-staging/watch

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "watch:311-strict" \
    --build-arg BUILD_STRICT=1 \
    --build-arg BASE_IMAGE=pyenv:311 \
    --build-arg PYTHON_VERSION=3.11.2 \
    -f ./dockerfiles/watch.Dockerfile .

docker run \
    --volume "$HOME/code/watch":/host-watch:ro \
    --runtime=nvidia -it watch:311-strict bash


git remote add dockerhost /host-watch/.git


   # Push the container to smartgitlab
IMAGE_NAME=watch:311-strict
docker tag $IMAGE_NAME registry.smartgitlab.com/kitware/$IMAGE_NAME
docker push registry.smartgitlab.com/kitware/$IMAGE_NAME

   # Will need to bake in a model

"
EOF
