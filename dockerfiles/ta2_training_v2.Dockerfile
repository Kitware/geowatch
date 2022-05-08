#FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ARG BUILD_STRICT=0


# Note:
# Nvidia updated the signing key
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# https://forums.developer.nvidia.com/t/failed-to-fetch-https-developer-download-nvidia-com-compute-machine-learning-repos-ubuntu1804-x86-64-packages-gz/156287/3

#sudo apt-key del 7fa2af80
#wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb

RUN echo "trying to fix nvidia stuff" && 
    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean 

#RUN echo "trying to fix nvidia stuff" && \
#    apt-key del 7fa2af80 && \
#    INST_ARCH=$(uname -m) && echo $INST_ARCH && \
#    NAME=$( (. /etc/os-release && echo "$NAME") | tr '[:upper:]' '[:lower:]' ) && \
#    VER=$( (. /etc/os-release && echo "$VERSION_ID") | sed 's/\.//g' ) && \
#    NVIDIA_DISTRO=${NAME}${VER} && \
#    NVIDIA_DISTRO_ARCH=${NVIDIA_DISTRO}/${INST_ARCH} && \
#    echo "NVIDIA_DISTRO_ARCH = $NVIDIA_DISTRO_ARCH" && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/"$NVIDIA_DISTRO_ARCH"/3bf863cc.pub

#wget "https://developer.download.nvidia.com/compute/cuda/repos/${NVIDIA_DISTRO_ARCH}/cuda-keyring_1.0-1_all.deb"
# sudo dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
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
        wget jq rsync \
        p7zip-full \
        unzip \
        ssh tmux tree curl iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py38_4.9.2
ARG CONDA_MD5=122c8c9beb51e124ab32a0fa6426c656

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_MD5}  miniconda.sh" > miniconda.md5 && \
    if ! md5sum --status -c miniconda.md5; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate watch" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /root
RUN echo $(pwd)

COPY conda_env.yml /root/code/watch/
COPY requirements /root/code/watch/requirements
COPY dev /root/code/watch/dev

RUN if [ "$BUILD_STRICT" -eq 1 ]; then \
    (cd /root/code/watch && ./dev/make_strict_req.sh && conda env create -f conda_env_strict.yml); \
else \
    (cd /root/code/watch && conda env create -f conda_env.yml); \
fi

COPY aws/dvc_and_aws.txt /root/code/watch/aws/dvc_and_aws.txt

# Fully strict DVC+AWS
RUN if [ "$BUILD_STRICT" -eq 1 ]; then \
    (cd /root/code/watch && conda activate watch && sed 's/>=/==/g' aws/dvc_and_aws.txt > aws/dvc_and_aws-strict.txt && pip install -r aws/dvc_and_aws-strict.txt); \
else \
    (cd /root/code/watch && conda activate watch && pip install -r aws/dvc_and_aws.txt); \
fi
# Has conflicts with DVC due to overly restrictive requirements
#RUN pip install awscli
#RUN conda activate watch && \
#    pip install dvc[s3]

COPY . /root/code/watch

RUN conda activate watch && \
    pip install --no-deps -e /root/code/watch

# docker build --build-arg BUILD_STRICT=1 .
