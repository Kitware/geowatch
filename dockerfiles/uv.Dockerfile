# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PIP_ROOT_USER_ACTION=ignore

# Install Prerequisites base tools, system Python (only needed for bootstrapping)
RUN <<EOF
#!/bin/bash
apt update -q

DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    curl wget git ca-certificates \
    python3 python3-venv python3-pip \
    build-essential \
    libgl1 libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender1

apt clean
rm -rf /var/lib/apt/lists/*
EOF


ARG PYTHON_VERSION=3.13

# Install uv
RUN <<EOF

set -e

# Install uv (from Astral)
curl -LsSf https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.cargo/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# Use uv to install Python 3.13 and seed venv
uv venv /env --python=$PYTHON_VERSION --seed


echo "Write bashrc and profile"
BASHRC_CONTENTS='
# Add the pyenv command to our environment if it exists
export HOME="/root"
export PATH="$HOME/.cargo/bin:$PATH"

# Auto-activate the venv on login
source /env/bin/activate

'
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile


EOF


################
### __DOCS__ ###
################
RUN <<EOF
echo '
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/

cd $HOME/code/geowatch
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t uv:313 \
    --build-arg PYTHON_VERSION=3.13 \
    -f ./dockerfiles/uv.Dockerfile .

docker run --gpus=all -it uv:313 bash

SeeAlso:
~/code/ci-docker/uv.Dockerfile

'
EOF



