# syntax=docker/dockerfile:1.5.0

# This dockerfile uses new-ish buildkit syntax. 
# Details on how to run are on the bottom of the file.
ARG BASE_IMAGE=gitlab.kitware.com:4567/computer-vision/ci-docker/uv:0.7.19-python3.13

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

ENV PATH="/root/.local/bin:${PATH}"


### Install AWS CLI (This might be handled by run_developer_setup soon)
RUN <<EOF

mkdir -p "$HOME/tmp/setup-aws"
cd "$HOME/tmp/setup-aws"

# Download the CLI tool for linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscli-exe-linux-x86_64.zip"

# Import the amazon GPG public key
echo "
    -----BEGIN PGP PUBLIC KEY BLOCK-----

    mQINBF2Cr7UBEADJZHcgusOJl7ENSyumXh85z0TRV0xJorM2B/JL0kHOyigQluUG
    ZMLhENaG0bYatdrKP+3H91lvK050pXwnO/R7fB/FSTouki4ciIx5OuLlnJZIxSzx
    PqGl0mkxImLNbGWoi6Lto0LYxqHN2iQtzlwTVmq9733zd3XfcXrZ3+LblHAgEt5G
    TfNxEKJ8soPLyWmwDH6HWCnjZ/aIQRBTIQ05uVeEoYxSh6wOai7ss/KveoSNBbYz
    gbdzoqI2Y8cgH2nbfgp3DSasaLZEdCSsIsK1u05CinE7k2qZ7KgKAUIcT/cR/grk
    C6VwsnDU0OUCideXcQ8WeHutqvgZH1JgKDbznoIzeQHJD238GEu+eKhRHcz8/jeG
    94zkcgJOz3KbZGYMiTh277Fvj9zzvZsbMBCedV1BTg3TqgvdX4bdkhf5cH+7NtWO
    lrFj6UwAsGukBTAOxC0l/dnSmZhJ7Z1KmEWilro/gOrjtOxqRQutlIqG22TaqoPG
    fYVN+en3Zwbt97kcgZDwqbuykNt64oZWc4XKCa3mprEGC3IbJTBFqglXmZ7l9ywG
    EEUJYOlb2XrSuPWml39beWdKM8kzr1OjnlOm6+lpTRCBfo0wa9F8YZRhHPAkwKkX
    XDeOGpWRj4ohOx0d2GWkyV5xyN14p2tQOCdOODmz80yUTgRpPVQUtOEhXQARAQAB
    tCFBV1MgQ0xJIFRlYW0gPGF3cy1jbGlAYW1hem9uLmNvbT6JAlQEEwEIAD4CGwMF
    CwkIBwIGFQoJCAsCBBYCAwECHgECF4AWIQT7Xbd/1cEYuAURraimMQrMRnJHXAUC
    ZMKcEgUJCSEf3QAKCRCmMQrMRnJHXCilD/4vior9J5tB+icri5WbDudS3ak/ve4q
    XS6ZLm5S8l+CBxy5aLQUlyFhuaaEHDC11fG78OduxatzeHENASYVo3mmKNwrCBza
    NJaeaWKLGQT0MKwBSP5aa3dva8P/4oUP9GsQn0uWoXwNDWfrMbNI8gn+jC/3MigW
    vD3fu6zCOWWLITNv2SJoQlwILmb/uGfha68o4iTBOvcftVRuao6DyqF+CrHX/0j0
    klEDQFMY9M4tsYT7X8NWfI8Vmc89nzpvL9fwda44WwpKIw1FBZP8S0sgDx2xDsxv
    L8kM2GtOiH0cHqFO+V7xtTKZyloliDbJKhu80Kc+YC/TmozD8oeGU2rEFXfLegwS
    zT9N+jB38+dqaP9pRDsi45iGqyA8yavVBabpL0IQ9jU6eIV+kmcjIjcun/Uo8SjJ
    0xQAsm41rxPaKV6vJUn10wVNuhSkKk8mzNOlSZwu7Hua6rdcCaGeB8uJ44AP3QzW
    BNnrjtoN6AlN0D2wFmfE/YL/rHPxU1XwPntubYB/t3rXFL7ENQOOQH0KVXgRCley
    sHMglg46c+nQLRzVTshjDjmtzvh9rcV9RKRoPetEggzCoD89veDA9jPR2Kw6RYkS
    XzYm2fEv16/HRNYt7hJzneFqRIjHW5qAgSs/bcaRWpAU/QQzzJPVKCQNr4y0weyg
    B8HCtGjfod0p1A==
    =gdMc
    -----END PGP PUBLIC KEY BLOCK-----
" | sed -e 's|^ *||' > aws2.pub
cat aws2.pub
gpg --import aws2.pub

# Set the trust level of the key
KEY_FPATH=aws2.pub
#KEY_ID=$(gpg --list-packets <"$KEY_FPATH" | awk '$1=="keyid:"{print$2;exit}')
KEY_ID=FB5DB77FD5C118B80511ADA8A6310ACC4672475C
echo "KEY_ID = $KEY_ID"
(echo 5; echo y; echo save) |
  gpg --command-fd 0 --no-tty --no-greeting -q --edit-key "$KEY_ID" trust

# Download the signature and verify the CLI tool is signed by amazon
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig" -o "awscli-exe-linux-x86_64.zip.sig"

gpg --verify awscli-exe-linux-x86_64.zip.sig awscli-exe-linux-x86_64.zip

# Unzip the downloaded installer
7z x awscli-exe-linux-x86_64.zip

# If you want to install somewhere else, change the PREFIX variable
PREFIX="$HOME/.local"
mkdir -p "$PREFIX"/bin
./aws/install --install-dir "$PREFIX/aws-cli" --bin-dir "$PREFIX/bin" --update

"$PREFIX"/bin/aws --version
EOF


WORKDIR /root
RUN mkdir -p /root/code

# Stage just enough of the geowatch source to run the build
# (this lets us modify supporting scripts while maintaining docker caches)
COPY setup.py               /root/code/geowatch/
COPY pyproject.toml         /root/code/geowatch/
COPY run_developer_setup.sh /root/code/geowatch/
COPY dev/make_strict_req.sh /root/code/geowatch/dev/make_strict_req.sh
COPY requirements           /root/code/geowatch/requirements
COPY geowatch               /root/code/geowatch/geowatch
COPY geowatch_tpl           /root/code/geowatch/geowatch_tpl

#RUN echo $(pwd)

ARG BUILD_STRICT=0

#SHELL ["/bin/bash", "--login", "-c"]

ARG DEV_TRACE=""

# Setup primary dependencies
# Note: special syntax for caching deps
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
#source $HOME/activate

echo "Preparing to pip install geowatch"

which python
which pip
python --version
pip --version
pwd
ls -altr
cd /root/code/geowatch
pwd
ls -altr

echo "Run GeoWATCH developer setup:"
WATCH_STRICT=$BUILD_STRICT WITH_MMCV=0 WITH_DVC=0 WITH_COLD=0 WITH_TENSORFLOW=0 WITH_AWS=1 WITH_COMPAT=1 WITH_APT_ENSURE=0 DEV_TRACE="$DEV_TRACE" bash run_developer_setup.sh

EOF



#### Copy over the rest of the repo structure
COPY .git          /root/code/geowatch/.git


# Run simple tests
RUN <<EOF
#!/bin/bash
#source $HOME/activate

echo "Start simple tests"
EAGER_IMPORT_MODULES=geowatch python -c "import geowatch; print(geowatch.__version__)"
EAGER_IMPORT_MODULES=geowatch python -m geowatch --help
EOF


# Remove the requirements folder we added so we can checkout the symlink
RUN rm -rf /root/code/geowatch/requirements

# Copy over the rest of the repo
COPY . /root/code/geowatch

WORKDIR /root/code/geowatch

RUN <<EOF
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/
echo "

    # SeeAlso:
    # ~/code/watch-smartflow-dags/prepare_system.sh

    # An invocation for basic end-to-end building is:

    # Build the geowatch image
    cd ~/code/geowatch
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "geowatch:uv0.7.19-python3.13-strict" \
        --build-arg BUILD_STRICT=1 \
        --build-arg DEV_TRACE=0 \
        -f ./dockerfiles/geowatch.Dockerfile .

    docker run \
        --volume "$HOME/code/geowatch":/host-geowatch:ro \
        --gpus=all -it geowatch:uv0.7.19-python3.13-strict bash

    IMAGE_VERSION=$(docker run --runtime=nvidia -it geowatch:311-strict python -c "import geowatch; print(geowatch.__version__)")
    IMAGE_VERSION=$(python -c "import geowatch; print(geowatch.__version__)")
    echo "IMAGE_VERSION=$IMAGE_VERSION"

    docker login gitlab.kitware.com:4567
    docker tag geowatch:311-strict gitlab.kitware.com:4567/computer-vision/geowatch:$IMAGE_VERSION-cp311-strict
    docker push gitlab.kitware.com:4567/computer-vision/geowatch:$IMAGE_VERSION-cp311-strict

    docker pull gitlab.kitware.com:4567/computer-vision/geowatch:$IMAGE_VERSION-cp311-strict

    docker tag gitlab.kitware.com:4567/computer-vision/geowatch:0.17.0-cp311-strict geowatch:0.17.0-cp311-strict

   # Will need to bake in a model
   # For futher instructions see: 
   # ../docs/source/manual/smartflow/smartflow_running_the_system.rst


    # Notes about running the images

    # host machine.
    LOCAL_WORK_DPATH=$HOME/temp/{temp_location}/ingress
    LOCAL_CODE_DPATH=$HOME/code
    LOCAL_DATA_DPATH=$HOME/data
    # Run the docker image
    mkdir -p "$LOCAL_WORK_DPATH"
    cd "$LOCAL_WORK_DPATH"
    docker run \
        --runtime=nvidia \
        --volume "$LOCAL_WORK_DPATH":/tmp/ingress \
        --volume "$LOCAL_CODE_DPATH":/extern_code:ro \
        --volume "$LOCAL_DATA_DPATH":/extern_data:ro \
        --volume "$HOME"/.aws:/root/.aws:ro \
        --volume "$HOME"/.cache/pip:/pip_cache \
        -it geowatch:0.17.0-cp311-strict bash

    git remote add host /extern_code/geowatch/.git
    git remote -v
    git fetch host
"
EOF
