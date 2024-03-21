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
import os
import pathlib
import shutil
import tempfile

import numpy as np

import model_navigator as nav


def get_dataloader():
    """Returns a random dataloader containing 10 batches of 1x5 tensors"""
    return [{"input__0": np.random.rand(1, 5).astype("float32")} for _ in range(10)]


def main():
    with tempfile.TemporaryDirectory() as tempdir:
        workspace = pathlib.Path(tempdir)

        """Get model, dataloader, and run optimization"""
        onnx_model = pathlib.Path("./model.onnx")

        dataloader = get_dataloader()

        """
        Optimizes ONNX to generate TensorRT plan
        """
        nav.onnx.optimize(
            model=onnx_model,
            dataloader=dataloader,
            target_formats=(nav.Format.TENSORRT,),
            workspace=workspace,
            optimization_profile=nav.OptimizationProfile(max_batch_size=1024),
            custom_configs=[nav.TensorRTConfig(precision=(nav.TensorRTPrecision.FP32,))],
            verbose=True,
        )

        src_path = workspace / "trt-fp32" / "model.plan"
        dst_path = pathlib.Path(os.getcwd()) / "model.plan"

        shutil.copyfile(src=src_path, dst=dst_path)


if __name__ == "__main__":
    main()
