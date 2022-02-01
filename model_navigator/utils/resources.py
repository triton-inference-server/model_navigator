# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from typing import NamedTuple

from model_navigator.model import Format
from model_navigator.triton.config import BackendAccelerator


class Resource:
    ONNX = "ONNX"
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    TENSORFLOW_TRT = "TensorRT in TensorFlow"
    TENSORRT = "TensorRT"
    TRITON_SERVER = "Triton Inference Server"
    AMP_ACCELERATOR = "Automatic FP16 Optimization"


class ResourceItem(NamedTuple):
    name: str
    link: str


FORMAT_RESOURCES = {
    Resource.ONNX: ResourceItem(name=Resource.ONNX, link="https://github.com/onnx"),
    Resource.PYTORCH: ResourceItem(name=Resource.PYTORCH, link="https://github.com/pytorch/pytorch"),
    Resource.TENSORFLOW: ResourceItem(name=Resource.TENSORFLOW, link="https://github.com/tensorflow/tensorflow"),
    Resource.TENSORRT: ResourceItem(name=Resource.TENSORRT, link="https://github.com/NVIDIA/TensorRT"),
    Resource.TENSORFLOW_TRT: ResourceItem(name=Resource.TENSORFLOW_TRT, link="https://github.com/tensorflow/tensorrt"),
}

TRITON_RESOURCES = {
    Resource.TRITON_SERVER: ResourceItem(
        name=Resource.TRITON_SERVER, link="https://github.com/triton-inference-server"
    ),
    Resource.ONNX: ResourceItem(
        name=f"{Resource.ONNX} Backend", link="https://github.com/triton-inference-server/onnxruntime_backend"
    ),
    Resource.PYTORCH: ResourceItem(
        name=f"{Resource.PYTORCH} Backend", link="https://github.com/triton-inference-server/pytorch_backend"
    ),
    Resource.TENSORFLOW: ResourceItem(
        name=f"{Resource.TENSORFLOW} Backend", link="https://github.com/triton-inference-server/tensorflow_backend"
    ),
}

FORMAT2RESOURCE = {
    Format.TF_SAVEDMODEL: Resource.TENSORFLOW,
    Format.TF_TRT: Resource.TENSORFLOW_TRT,
    Format.ONNX: Resource.ONNX,
    Format.TENSORRT: Resource.TENSORRT,
    Format.TORCHSCRIPT: Resource.PYTORCH,
    Format.TORCH_TRT: Resource.PYTORCH,
}

ACCELERATOR2RESOURCE = {
    BackendAccelerator.TRT: Resource.TENSORRT,
    BackendAccelerator.AMP: Resource.AMP_ACCELERATOR,
}

FORMAT_RESOURCE = {key: FORMAT_RESOURCES[value] for key, value in FORMAT2RESOURCE.items()}
