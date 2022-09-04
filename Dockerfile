FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

#############################################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#############################################################

#############################################################
# You should modify this to match your CPU compute capability
ENV MAX_JOBS=2
ENV OMP_NUM_THREADS=16
#############################################################

ENV CUDA_HOME /usr/local/cuda-11.3
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install base utilities
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install keyboard-configuration
RUN apt-get -y upgrade
RUN apt-get -y install build-essential git-all libopenblas-dev mate-desktop-environment-core ninja-build openssh-server wget tmux xauth xterm
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda (we need Python 3.8+ to get SciPy 1.8.0+)
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh 
ENV PATH=$CONDA_DIR/bin:$PATH
RUN . "$CONDA_DIR/bin/activate"

# Install PyTorch
RUN conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install MinkowskiEngine
RUN conda install -y openblas-devel -c anaconda
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--blas=openblas"

# Install ConformalLayers and all dependencies
RUN pip install -U git+https://github.com/Prograf-UFF/ConformalLayers -v