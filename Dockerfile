FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates git libgl1-mesa-glx

# To update to a newer version see:
# https://docs.conda.io/en/latest/miniconda_hashes.html for updating
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate watch" >> ~/.bashrc && \
    /opt/conda/bin/conda config --set channel_priority flexible && \
    /opt/conda/bin/conda update -n base conda -y && \
    /opt/conda/bin/conda clean -afy

SHELL ["/bin/bash", "--login", "-c"]

COPY . /watch

RUN conda env create -f /watch/conda_env.yml

RUN conda activate watch && \
    pip install --no-deps -e /watch
