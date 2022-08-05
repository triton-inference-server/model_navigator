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

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    nav_workdir = Path(args.workdir)
    for model_name, model_cls in MODELS.items():
        nav.LOGGER.info(f"Testing {model_name}...")
        model = model_cls(pretrained=True).eval()
        pkg_desc = nav.torch.export(
            model=model,
            model_name=model_name + "_pyt",
            dataloader=DATALOADER,
            workdir=nav_workdir,
            target_device="cuda",
            profiler_config=nav.ProfilerConfig(measurement_request_count=200),
        )
        if model_name == "se-resnext101-32x4d":
            expected_runtimes = {
                "torchscript-script": [nav.RuntimeProvider.PYT.value],
                "torchscript-trace": [nav.RuntimeProvider.PYT.value],
                "trt-fp32": [nav.RuntimeProvider.TRT.value],
                "trt-fp16": [nav.RuntimeProvider.TRT.value],
                "onnx": [nav.RuntimeProvider.CPU.value, nav.RuntimeProvider.CUDA.value],
            }
        else:
            expected_runtimes = {
                "torchscript-script": [nav.RuntimeProvider.PYT.value],
                "torchscript-trace": [nav.RuntimeProvider.PYT.value],
                "onnx": [nav.RuntimeProvider.CPU.value, nav.RuntimeProvider.CUDA.value],
                "torch-trt-trace-fp16": [nav.RuntimeProvider.PYT.value],
                "torch-trt-trace-fp32": [nav.RuntimeProvider.PYT.value],
                "torch-trt-script-fp16": [nav.RuntimeProvider.PYT.value],
                "torch-trt-script-fp32": [nav.RuntimeProvider.PYT.value],
                "trt-fp32": [nav.RuntimeProvider.TRT.value],
                "trt-fp16": [nav.RuntimeProvider.TRT.value],
            }
        nav.LOGGER.info(f"{pkg_desc.get_formats_status()=}")
        for format, runtimes_status in pkg_desc.get_formats_status().items():
            for runtime, status in runtimes_status.items():
                if runtime in expected_runtimes.get(format, {}):
                    assert (
                        status == nav.Status.OK
                    ), f"{format} {runtime} status is {status}, but expected runtimes are {expected_runtimes}."
                else:
                    if status == nav.Status.OK:
                        nav.LOGGER.warning(f"{format} {runtime} status is {status} but it is not in expected runtimes.")
        nav.save(pkg_desc, nav_workdir / f"{model_name}_pyt.nav")

        nav.LOGGER.info(f"{model_name} passed.")
    nav.LOGGER.info("All models passed.")
