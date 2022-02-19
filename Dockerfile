ARG PYTORCH="1.10.2"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG WANDB_KEY

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#############################################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#############################################################

#############################################################
# You should modify this to match your CPU compute capability
ENV MAX_JOBS=2
#############################################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --install-option="--force_cuda" --install-option="--blas=openblas"
RUN pip install -U git+https://github.com/Prograf-UFF/ConformalLayers -v

RUN printf "machine api.wandb.ai\n\tlogin user\n\tpassword $WANDB_KEY" >> /root/.netrc
