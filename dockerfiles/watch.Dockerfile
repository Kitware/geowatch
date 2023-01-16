# syntax=docker/dockerfile:1.3.0-labs

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.
FROM pyenv:310

ENV HOME=/root

## Install Prerequisites 
RUN <<EOF
#!/bin/bash
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
        ffmpeg tmux jq tree p7zip-full rsync
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


# Stage the watch source
COPY setup.py /watch/
COPY pyproject.toml /watch/
COPY requirements /watch/requirements
COPY watch /watch/watch

SHELL ["/bin/bash", "--login", "-c"]

RUN echo $(pwd)

ARG BUILD_STRICT=0

# Setup primary dependencies
RUN <<EOF
#!/bin/bash
source $HOME/activate

# Always use the latest Python build tools
python -m pip install pip setuptools wheel build -U

if [ "$BUILD_STRICT" -eq 1 ]; then
    echo "BUILDING STRICT VARIANT"
    pip install -e /watch[runtime-strict,development-strict,optional-strict,headless-strict]
else
    echo "BUILDING LOOSE VARIANT"
    pip install -e /watch[development,optional,headless]
    # python -m pip install dvc[all]>=2.13.0
    # pip install awscli
fi

EOF


# Finalize more fickle dependencies
RUN <<EOF
#!/bin/bash
source $HOME/activate

cd /watch
if [ "$BUILD_STRICT" -eq 1 ]; then
    echo "FINALIZE STRICT VARIANT DEPS"
    sed 's/>=/==/g' requirements/gdal.txt > requirements/gdal-strict.txt
    pip install -r requirements/gdal-strict.txt
else
    echo "FINALIZE LOOSE VARIANT DEPS"
    pip install -r requirements/gdal.txt
fi

EOF


# Install other useful tools
RUN <<EOF
#!/bin/bash
source $HOME/activate

# python -m pip install dvc[all]>=2.13.0
# pip install scikit-image>=0.18.1
pip install awscli

EOF


# Run simple tests
RUN <<EOF
#!/bin/bash
source $HOME/activate

echo "Start simple tests"
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m watch --help
EOF

# Copy over the rest of the repo
COPY . /watch

RUN <<EOF
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
echo "

# docker login
# docker pull docker/dockerfile:1.3.0-labs

cd $HOME/code/watch

# Either build the pyenv image or
docker pull gitlab.kitware.com:4567/smart/watch/pyenv:310
docker tag gitlab.kitware.com:4567/smart/watch/pyenv:310 pyenv:310 

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "watch:310" \
    --build-arg PYTHON_VERSION=3.10.5 \
    -f ./dockerfiles/watch.Dockerfile .

docker run --runtime=nvidia -it watch:310 bash

"
EOF
