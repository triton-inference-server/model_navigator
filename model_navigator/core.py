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

import re
from enum import Enum


class Parameter(Enum):
    def __lt__(self, other: "Parameter") -> bool:
        return self.value < other.value


class Accelerator(Parameter):
    NONE = "none"
    AMP = "amp"
    CUDA = "cuda"
    TRT = "trt"


class Precision(Parameter):
    ANY = "any"
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"


# TODO: review of format - maybe some of them are redundant
class Format(Parameter):
    TF_GRAPHDEF = "tf-graphdef"
    TF_CHECKPOINT = "tf-checkpoint"
    TF_SAVEDMODEL = "tf-savedmodel"
    TF_TRT = "tf-trt"
    TF_ESTIMATOR = "tf-estimator"
    TF_KERAS = "tf-keras"
    ONNX = "onnx"
    TRT = "trt"
    TS_SCRIPT = "ts-script"
    TS_TRACE = "ts-trace"
    PYT = "pyt"


class Container(NamedTuple):
    image: str = str()
    tag: str = str()

    @classmethod
    def container_image(cls, version: str):
        if version is not None and not re.match("^[1-9][0-9].[0-9]{2}([.][0-9])*$", version):
            raise ValueError(
                f"""Invalid container version: {version}. Please provide in following format are allowed: """
                """20.06, 20.06.1"""
            )

        return f"{cls.image}:{version}-{cls.tag}"
