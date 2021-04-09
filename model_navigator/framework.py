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
from .core import Container


class Framework(Container):
    name: str
    acronym: str


class PyTorch(Framework):
    name = "PyTorch"
    acronym = "PyT"
    image = "nvcr.io/nvidia/pytorch"
    tag = "py3"


class TensorFlow1(Framework):
    name = "TensorFlow1"
    acronym = "TF1"
    image = "nvcr.io/nvidia/tensorflow"
    tag = "tf1-py3"


class TensorFlow2(Framework):
    name = "TensorFlow2"
    acronym = "TF2"
    image = "nvcr.io/nvidia/tensorflow"
    tag = "tf2-py3"
