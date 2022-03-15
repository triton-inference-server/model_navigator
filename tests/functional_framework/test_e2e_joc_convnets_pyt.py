#!/usr/bin/env python3
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

import tempfile
from pathlib import Path

import torch  # pytype: disable=import-error
from image_classification import models as convnet_models  # pytype: disable=import-error

import model_navigator as nav

MODELS = {
    "resnet50": convnet_models.resnet50,
    "resnext101-32x4d": convnet_models.resnext101_32x4d,
    "se-resnext101-32x4d": convnet_models.se_resnext101_32x4d,
    "efficientnet-b0": convnet_models.efficientnet_b0,
    "efficientnet-b4": convnet_models.efficientnet_b4,
    "efficientnet-widese-b0": convnet_models.efficientnet_widese_b0,
    "efficientnet-widese-b4": convnet_models.efficientnet_widese_b4,
    # "efficientnet-quant-b0": convnet_models.efficientnet_quant_b0,
    # "efficientnet-quant-b4": convnet_models.efficientnet_quant_b4
}


DATALOADER = [torch.randn(1, 3, 224, 224)]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        nav_workdir = Path(tmp_dir) / "navigator_workdir"
        for model_name, model_cls in MODELS.items():
            nav.LOGGER.info(f"Testing {model_name}...")
            model = model_cls(pretrained=True).eval()
            pkg_desc = nav.torch.export(
                model=model,
                dataloader=DATALOADER,
                workdir=nav_workdir,
                target_device="cuda",
            )
            expected_formats = (
                "torchscript-script",
                "torchscript-trace",
                "onnx",
                "torch-trt-script",
                "torch-trt-trace",
                "trt-fp32",
                "trt-fp16",
            )
            for format, status in pkg_desc.get_formats_status().items():
                status = list(status.values())[0]
                assert (status == nav.Status.OK) == (
                    format in expected_formats
                ), f"{format} status is {status.value}, but expected formats are {expected_formats}."

            nav.LOGGER.info(f"{model_name} passed.")
    nav.LOGGER.info("All models passed.")
