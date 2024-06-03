# syntax = docker/dockerfile:1.2
FROM nvidia/cuda:12.1.0-base-ubuntu20.04
# ^ make sure matches cuda version below (12.1)

# NOTE: this Dockerfile needs to be build from the repo root directory! e.g. run:
# docker build -f k8s/Dockerfile -t kingb12/nc_latent_tod:latest .
# rather than from this directory:
# docker build -f Dockerfile -t kingb12/nc_latent_tod:latest .


# =================== Project Agnostic Steps: Install Python, Conda, Zsh ===================

# Remove any third-party apt sources to avoid issues with expiring keys. Then, install some basic utilities and python-dev
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y \
    curl \
    zsh \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    wget \
    libx11-6 \
    python-dev \
 && rm -rf /var/lib/apt/lists/*

# work from the home directory
WORKDIR /root

# Install and use ZSH: optional, but very useful IMO. Default powerline10k theme, no plugins installed, and set shell to zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" && \
    chsh -s /usr/bin/zsh root

# Setup and install miniconda
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash "Miniconda3-latest-Linux-x86_64.sh" -b -p "/root/miniconda" && \
    miniconda/bin/conda init --all


# some final setup: add this lint to the top of each shell file to turn off output in interactive modes
# this is required for SFTP to work
RUN echo "[[ "$-" != *i* ]] && return" > .bashrc && touch ~/.zshrc && sed -i '1s/^/[[ "$-" != *i* ]] \&\& return\n/' .zshrc

# =================== Project Specific Steps: Install Dependencies contained in setup.cfg ===================

# Add miniconda to the path, includes `conda` command itself
ENV PATH="/root/miniconda/bin:${PATH}"

# Create a conda environment for the project w/ Python 3.10
RUN --mount=type=cache,target=/opt/conda/pkgs conda init zsh && conda create python=3.10 --prefix venv

# conda init is needed before activation, but only takes effect after shell restart. Since each new Dockerfile line
# runs in a new shell instance, this doesn't actually take effect as expected. Instead, we'll change our shell to the
# one installed in the conda environment itself. Make RUN commands use the new environment:
SHELL ["conda", "run", "-p", "/root/venv", "/bin/bash", "-c"]


# These can speed up builds that include conda/pip install (like below)
ENV PIP_CACHE_DIR .cache/buildkit/pip
RUN mkdir -p $PIP_CACHE_DIR

# Install PyTorch and related libraries directly from conda
RUN --mount=type=cache,target=/opt/conda/pkgs conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda install -c anaconda gxx_linux-64 nvidia::cuda-nvcc

# create and enter the project directory
RUN mkdir nc_latent_tod
WORKDIR "/root/nc_latent_tod"

# Add in the project code, install some tricky dependencies, then install the project itself and remaining dependencies in -e mode
COPY . .
# Make sure to do flash-attn last! Somehow it doesn't cooperate with transformers/accelerate otherwise
RUN --mount=type=cache,target=.cache/buildkit/pip pip install pyzmq faiss-cpu faiss-gpu && pip install packaging ninja==1.10.2 && pip install --user -e . && pip install flash-attn --no-build-isolation

# on start up, switch to zsh! k8s jobs will skip this command and run the command in the job spec, so we shouldn't put anything important here
CMD ["zsh"]
