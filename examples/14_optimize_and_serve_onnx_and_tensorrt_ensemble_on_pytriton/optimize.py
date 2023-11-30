#!/usr/bin/env python3
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
import pathlib

import numpy as np

import model_navigator as nav


def get_dataloader():
    """Returns a ramdom dataloader containing 10 batches of 1x5 tensors"""
    return [{"input__0": np.random.rand(1, 5).astype("float32")} for _ in range(10)]


def main():
    """Get model, dataloader, and run optimization"""
    onnx_model = pathlib.Path("./model.onnx")
    tensorrt_model = pathlib.Path("./model.plan")

    dataloader = get_dataloader()

    """
    Optimizes ONNX and TensorRT models.
    """

    onnx_package = nav.onnx.optimize(
        model=onnx_model,
        dataloader=dataloader,
        target_formats=(nav.Format.ONNX,),
        workspace=pathlib.Path("onnx_workspace"),
        optimization_profile=nav.OptimizationProfile(max_batch_size=1024),
    )

    nav.package.save(onnx_package, "onnx_linear.nav", override=True)

    tensorrt_package = nav.tensorrt.optimize(
        model=tensorrt_model,
        dataloader=dataloader,
        workspace=pathlib.Path("tensorrt_workspace"),
        optimization_profile=nav.OptimizationProfile(max_batch_size=1024),
    )

    nav.package.save(tensorrt_package, "tensorrt_linear.nav", override=True)


if __name__ == "__main__":
    main()
