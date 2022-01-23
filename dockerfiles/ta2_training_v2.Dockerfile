FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

ARG BUILD_STRICT=0

RUN apt-get update -q && \
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
        wget \
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
    (cd /root/code/watch/ && ./dev/make_strict_req.sh && conda env create -f conda_env_strict.yml); \
else \-
    (cd /root/code/watch/ && conda env create -f conda_env.yml); \
fiore--

RUN pip install awscli

COPY . /root/code/watch

RUN conda activate watch && \
    pip install --no-deps -e /watch

RUN conda activate watch && \
    pip install dvc[s3]

# docker build --build-arg BUILD_STRICT=1 .
