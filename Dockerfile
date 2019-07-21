# Runtime image
FROM ceshine/pytorch-apex-cuda:latest AS build

# Instal basic utilities
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends build-essential && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

COPY ./pytorch-pretrained-BERT-master.zip /tmp
COPY ./helperbot.zip /tmp
ARG PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
# ARG PIP_MIRROR=https://pypi.python.org/simple
RUN  pip install -i $PIP_MIRROR --upgrade pip && \
    pip install -i $PIP_MIRROR python-telegram-bot && \
    pip install -i $PIP_MIRROR /tmp/helperbot.zip && \
    pip install -i $PIP_MIRROR /tmp/pytorch-pretrained-BERT-master.zip && \
    rm -rf ~/.cache/pip

# runtime
from nvidia/cuda:10.0-base

ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo p7zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH $CONDA_DIR/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

RUN mkdir -p /opt/conda/

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME

COPY --chown=1000 --from=build /opt/conda/. $CONDA_DIR
