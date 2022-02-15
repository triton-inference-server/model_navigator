# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.01-py3
FROM $BASE_IMAGE

# DCGM version to install for Model Analyzer
ENV DCGM_VERSION=2.2.9
ENV MODEL_NAVIGATOR_CONTAINER=1

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Install Docker, NVIDIA Docker and DCGM
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common curl python3-dev python3-pip python-is-python3 libb64-dev wget git wkhtmltopdf && \
    \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian buster stable" && \
    apt-get update && \
    apt-get install --no-install-recommends -y docker-ce docker-ce-cli containerd.io && \
    \
    . /etc/os-release && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey| apt-key add - && \
    curl -s -L "https://nvidia.github.io/nvidia-docker/${ID}${VERSION_ID}/nvidia-docker.list" > /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y nvidia-docker2 && \
    \
    wget --progress=dot:giga https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get install -y datacenter-gpu-manager=1:${DCGM_VERSION} && \
    \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    mkdir -p /opt/model-navigator

# WAR for PEP660
RUN pip install --no-cache-dir --upgrade pip==21.2.4 setuptools==57.4.0

WORKDIR /opt/model-navigator
COPY . /opt/model-navigator
RUN pip3 install --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir -e .

ENTRYPOINT []
