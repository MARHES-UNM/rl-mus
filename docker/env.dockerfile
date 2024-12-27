# Ubuntu as our base OS
ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

# LABEL about the custom image
LABEL maintainer="gno320@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
    running linux env on Windows."

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# For creating non-root users
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEV_WORKSPACE=/home/$USERNAME/workspace
ARG CONDA_DIR=/home/$USERNAME/conda

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    # gives the user ability to install software
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Custom bash prompt via kirsle.net/wizards/ps1.html
# https://ss64.com/bash/syntax-prompt.html
RUN echo 'PS1="[$(echo -e "\xF0\x9F\x92\xB0")\[$(tput setaf 1)\]\u\[$(tput setaf 3)\]@\[$(tput setaf 2)\]\h\[$(tput sgr0)\]:\[$(tput setaf 6)\]\w\[$(tput setaf 3)\]$(git branch 2> /dev/null | sed -e "/^[^*]/d" -e "s/* \(.*\)/(\1)/")\[$(tput sgr0)\]]\\$ \[$(tput sgr0)\]"' >> /home/$USERNAME/.bashrc && \
    echo 'alias ls="ls --color=auto"' >> /home/$USERNAME/.bashrc 

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -tipsy && \
    ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc && \
    echo "conda activate base" >> /home/${USERNAME}/.bashrc

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# # install python3.8 and set it as default
# RUN apt-get update -y && \
#     apt install --no-install-recommends -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa -y && \ 
#     apt-get update -y && \
#     apt-get install --no-install-recommends -y \
#     python3.8 \
#     python3-dev && \
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt clean

# # additional dependencies
# RUN apt-get update -y && \
#     apt-get install --no-install-recommends -y \
#     qt5-default \
#     python3-pip \
#     python3-setuptools \
#     build-essential \
#     ffmpeg \
#     curl && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt clean

# # # Some TF tools expect a "python" binary
# RUN ln -s $(which python3.8) /usr/local/bin/python

# # [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

# # Install dependencies for the environment
# RUN python -m pip install --upgrade pip && \
#     python -m pip --no-cache-dir install --upgrade \
#     setuptools \
#     numpy \
#     Pillow \
#     matplotlib \
#     cycler \
#     tensorflow==2.6 \
#     "gym<0.20,>=0.17" \
#     pybullet \
#     stable_baselines3 \
#     'ray[rllib]' \
#     pyqt5

# # WORKDIR ${DEV_WORKSPACE}/
# COPY --chown=${USERNAME}:${USERNAME} gym-pybullet-drones ${DEV_WORKSPACE}/gym-pybullet-drones
# RUN cd ${DEV_WORKSPACE}/gym-pybullet-drones && \
#     python -m pip install -e .

# COPY docker/entrypoint.sh /
# ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]