# Our base OS Image
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# start with base image
FROM ${BASE_IMAGE}

# set python version to use
ENV PYTHON_VERSION=python3.11
# ARG CONDA_ENV=py39_gpu

# FROM directive resets ARG
ARG BASE_IMAGE

# LABEL about the custom image
LABEL maintainer="gno320@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
    running linux env on Windows."

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# For creating non-root users
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEV_WORKSPACE=/home/$USERNAME/workspace
ARG CONDA_DIR=/home/$USERNAME/conda

# Set timezone, options are America/Los_Angeles
ENV TZ=America/New_York

# adds anaconda to path
ENV PATH "${CONDA_DIR}/bin:$PATH"

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && apt-get update -y \
    && apt-get install -y sudo tzdata \
    # give the user ability to install software
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Custom bash prompt via kirsle.net/wizards/ps1.html
# https://ss64.com/bash/syntax-prompt.html
COPY docker/config/.vimrc /home/${USERNAME}/.
COPY docker/config/.bashrc_extend /home/${USERNAME}/.bashrc_extend
COPY docker/config/.gitconfig /home/${USERNAME}/.

# Appends custom functions to bashrc
RUN cat /home/${USERNAME}/.bashrc_extend >> /home/${USERNAME}/.bashrc \
    && rm /home/${USERNAME}/.bashrc_extend

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER ${USER_UID}
ENV HOME=/home/${USERNAME}
WORKDIR ${DEV_WORKSPACE}

# Install dev environment
RUN sudo apt-get update --fix-missing -y \
    && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    g++ \ 
    gcc \
    git \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    libjemalloc-dev \
    python-opengl \
    qt5-default \
    unrar \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    $(if [ "$AUTOSCALER" = "autoscaler" ]; then echo \
    tmux \
    screen \
    rsync \
    openssh-client \
    gnupg; fi) \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Install python
# https://linuxize.com/post/how-to-install-python-3-9-on-ubuntu-20-04/
# pip3.9 --version && \
RUN sudo apt update && \
    sudo apt install -y software-properties-common && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \ 
    sudo apt install -y \ 
    ${PYTHON_VERSION} \
    ${PYTHON_VERSION}-dev \
    ${PYTHON_VERSION}-distutils \
    ${PYTHON_VERSION}-venv \
    ${PYTHON_VERSION}-tk && \
    ${PYTHON_VERSION} --version && \
    ${PYTHON_VERSION} -m ensurepip && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 1 && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2 && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON_VERSION} 3 && \
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_VERSION} 1 && \
    # sudo update-alternatives --set python /usr/bin/${PYTHON_VERSION} && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# COPY ../requirements_gpu.txt /home/${USERNAME}/.
COPY ../clean_requirements.txt /home/${USERNAME}/.
RUN sudo chown -R dev:dev /home/${USERNAME} \
    && ${PYTHON_VERSION} -m pip install -r /home/${USERNAME}/clean_requirements.txt --timeout=100

# apt install -y \
#         software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt install -y \
#         ${PYTHON_VERSION} \
#         ${PYTHON_VERSION}-distutils \
#         ${PYTHON_VERSION}-venv && \
#     ${PYTHON_VERSION} --version && \
#     ${PYTHON_VERSION} -m ensurepip && \
#     pip3.9 --version



# # RUN wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
# RUN wget --quie "https://repo.anaconda.com/miniconda/Miniconda3-py39_24.11.1-0-Linux-x86_64.sh" \
#     -O ${HOME}/miniconda.sh \
#     && /bin/bash ${HOME}/miniconda.sh -b -u -p ${CONDA_DIR} \
#     && ${CONDA_DIR}/bin/conda init \ 
#     # && echo "export PATH=${CONDA_DIR}/bin:$PATH" >> /home/${USERNAME}/.bashrc \
#     && rm ${HOME}/miniconda.sh \
#     && ${CONDA_DIR}/bin/conda install -y \
#     libgcc python=${PYTHON_VERSION} \
#     # conda cleaning option
#     # -t or --tarballs
#     # -i or --index-cache
#     # -p or --packages
#     # -s or ... eh that was not documented, it probably relates to --source-cache and this issue
#     # -y or --yes
#     && ${CONDA_DIR}/bin/conda clean -tip -y --all \
#     # && sudo ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     # echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc && \
#     # echo "conda activate base" >> /home/${USERNAME}/.bashrc \
#     && ${CONDA_DIR}/bin/pip install --no-cache-dir \
#     flatbuffers \
#     cython==0.29.26 \
#     # Necessary for Dataset to work properly.
#     numpy\>=1.20 \
#     psutil \
#     # To avoid the following error on Jenkins:
#     # AttributeError: 'numpy.ufunc' object has no attribute '__module__'
#     && ${CONDA_DIR}/bin/pip uninstall -y dask \ 
#     # We install cmake temporarily to get psutil
#     && sudo apt-get autoremove -y cmake zlib1g-dev \
#     # We keep g++ on GPU images, because uninstalling removes CUDA Devel tooling
#     $(if [ "$BASE_IMAGE" = "ubuntu:focal" ]; then echo \
#     g++; fi) \
#     # Either install kubectl or remove wget 
#     && (if [ "$AUTOSCALER" = "autoscaler" ]; \
#     then wget -O - -q https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
#     && sudo touch /etc/apt/sources.list.d/kubernetes.list \
#     && echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list \
#     && sudo apt-get update \
#     && sudo apt-get install kubectl; \
#     else sudo apt-get autoremove -y wget; \
#     fi;) \
#     && sudo rm -rf /var/lib/apt/lists/* \
#     && sudo apt-get clean

# COPY ../environment.yml /home/${USERNAME}/.

# # RUN ${CONDA_DIR}/bin/pip --no-cache-dir install --upgrade pip \
#     # && ${CONDA_DIR}/bin/pip --no-cache-dir install -r /home/${USERNAME}/requirements.txt \
#     # && ${CONDA_DIR}/bin/conda env create --name ${CONDA_ENV} --file /home/${USERNAME}/environment.yml \
# RUN ${CONDA_DIR}/bin/conda env create --name ${CONDA_ENV} --file /home/${USERNAME}/environment.yml \
#     # && echo "conda activate ${CONDA_ENV}" >> /home/${USERNAME}/.bashrc \
#     && if [ $(python -c 'import sys; print(sys.version_info.minor)') != "6" ]; then \
#     ${CONDA_DIR}/bin/pip uninstall dataclasses typing -y; fi 

# set up lightweight display, -E,-preserve-env ensures we get environment variables when using sudo
RUN sudo apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive sudo -E apt-get install -y x11vnc \
    xvfb \
    xfce4 \
    fonts-wqy-microhei fonts-wqy-zenhei \
    && sudo apt-get autoremove \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* \
    # create X11 socket directory
    && install -d -m 1777 /tmp/.X11-unix \
    && mkdir -p ~/.vnc \
    && x11vnc -storepasswd 1234 ~/.vnc/passwd \
    # Fix error with dubious workspace: Git detect dubious ownership in repository
    && sudo chown ${USERNAME} ${DEV_WORKSPACE}

# A script for loading the GUI app
COPY docker/entrypoint.sh /entrypoint.sh

# Starts the app. 
ENTRYPOINT ["/entrypoint.sh"]