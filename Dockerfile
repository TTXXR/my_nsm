# NAME NSM MLP_EA
# Time 2021/07/29
# VERSION 0.1.1
FROM devnet.hub.woa.com/gameai/anim-torch-gpu:latest
LABEL maintainer="alantxren@tencent.com"
ENV PATH="/root/bin:${PATH}"
ENV PATH="/opt/anaconda2/bin:${PATH}"
ENV PATH="/usr/local/nvidia/bin:${PATH}"
ENV PYTHONPATH="/home/gameai/animation"
ENV BLADE_AUTO_UPGRADE="no"
ADD . /home/gameai/animation
WORKDIR /home/gameai/animation
