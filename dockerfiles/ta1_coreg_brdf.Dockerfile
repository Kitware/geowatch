FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ARG BUILD_STRICT=0

ARG DEBIAN_FRONTEND=noninteractive

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
        unzip \
        wget \
        xorg \
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

RUN mkdir -p /opt/bin

ARG L8_ANGLES_GEN_SHA256=20a8eae551cd48b4ce15ad7ce0991a45c7c1b7f15ac8baa7a3a9ec3aadda5c31
RUN wget --quiet https://landsat.usgs.gov/sites/default/files/documents/L8_ANGLES_2_7_0.tgz -O L8_ANGLES.tgz && \
    echo "${L8_ANGLES_GEN_SHA256}  L8_ANGLES.tgz" > L8_ANGLES.checksum && \
    if ! shasum -a 256 --status -c L8_ANGLES.checksum; then exit 1; fi && \
    tar xzf L8_ANGLES.tgz && \
    cd l8_angles && make && cp l8_angles /opt/bin/

ARG LANDSAT_ANGLES_SHA256=1b1c146b3305ba91570fdd05ab7059d382f67ba8f19952d3ea21d60efb5c6da7
RUN wget --quiet http://landsat.usgs.gov/sites/default/files/documents/LANDSAT_ANGLES_15_3_0.tgz -O LANDSAT_ANGLES.tgz && \
    echo "${LANDSAT_ANGLES_SHA256}  LANDSAT_ANGLES.tgz" > LANDSAT_ANGLES.checksum && \
    if ! shasum -a 256 --status -c LANDSAT_ANGLES.checksum; then exit 1; fi && \
    tar xzf LANDSAT_ANGLES.tgz && \
    cd landsat_angles && make && cp landsat_angles /opt/bin/

ARG MATLAB_SHA256=d67a6a2977a9119bd42d547126f6090b081b92610eff0e57071726244c06c593

RUN mkdir /mcr-install && \
    mkdir /opt/mcr && \
    cd /mcr-install && \
    wget --quiet https://ssd.mathworks.com/supportfiles/downloads/R2019b/Release/8/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019b_Update_8_glnxa64.zip && \
    echo "${MATLAB_SHA256}  MATLAB_Runtime_R2019b_Update_8_glnxa64.zip" > matlab.sha256 && \
if ! sha256sum --status -c matlab.sha256; then exit 1; fi && \
    unzip -q MATLAB_Runtime_R2019b_Update_8_glnxa64.zip && \
    rm -f MATLAB_Runtime_R2019b_Update_8_glnxa64.zip && \
    ./install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
    cd / && \
    rm -rf mcr-install

# Configure environment variables for MCR
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/mcr/v97/runtime/glnxa64:/opt/mcr/v97/bin/glnxa64:/opt/mcr/v97/sys/os/glnxa64:/opt/mcr/v97/sys/opengl/lib/glnxa64
ENV XAPPLRESDIR="/opt/mcr/v97/X11/app-defaults"

SHELL ["/bin/bash", "--login", "-c"]

RUN echo $(pwd)

COPY conda_env.yml /watch/
COPY requirements /watch/requirements
COPY dev /watch/dev

RUN if [ "$BUILD_STRICT" -eq 1 ]; then \
    (cd /watch && ./dev/make_strict_req.sh && conda env create -f conda_env_strict.yml); \
else \
    (cd /watch && conda env create -f conda_env.yml); \
fi

RUN pip install awscli

COPY . /watch

COPY ./matlab_binaries/main_BRDF_SingleBand /opt/bin/
COPY ./matlab_binaries/mainMTRA_HLS /opt/bin/

RUN conda activate watch && \
    pip install --no-deps -e /watch

ENV PATH $PATH:/opt/bin

# docker build --build-arg BUILD_STRICT=1 .
