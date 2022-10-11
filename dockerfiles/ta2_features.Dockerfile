FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ARG BUILD_STRICT=0

RUN echo "trying to fix nvidia stuff" && \
    rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean 

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py39_4.12.0
ARG CONDA_SHA256=78f39f9bae971ec1ae7969f0516017f2413f17796670f7040725dd83fcff5689

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_SHA256}  miniconda.sh" > miniconda.sha256 && \
    if ! sha256sum --status -c miniconda.sha256; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.sha256 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate watch" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

COPY conda_env.yml /watch/
COPY requirements /watch/requirements
COPY dev /watch/dev

RUN if [ "$BUILD_STRICT" -eq 1 ]; then \
    (cd /watch && ./dev/make_strict_req.sh && conda env create -f conda_env_strict.yml); \
else \
    (cd /watch && conda env create -f conda_env.yml); \
fi

RUN pip install awscli

COPY ./models /models

COPY ./watch /watch/watch
COPY ./scripts /watch/scripts
COPY ./setup.py /watch/setup.py

SHELL ["/bin/bash", "--login", "-c"]

RUN conda activate watch && \
    pip install --no-deps -e /watch

# docker build --build-arg BUILD_STRICT=1 .
