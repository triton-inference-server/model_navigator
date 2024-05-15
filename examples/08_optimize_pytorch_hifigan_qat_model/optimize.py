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
import argparse

import numpy as np
import torch  # pytype: disable=import-error
from nemo.collections.tts.models import HifiGanModel  # pytype: disable=import-error

import model_navigator as nav


class TTSWrapper(torch.nn.Module):
    def __init__(self, model, with_z=True) -> None:
        super().__init__()
        self.model = model
        self.with_z = with_z

    def forward(self, spec, z=None):
        if self.with_z:
            return self.model.forward_for_export(spec, z)
        else:
            return self.model.forward_for_export(spec)


def hifigan_qat_dataloader(with_z=True):
    input_dict = {"spec": np.random.randn(1, 80, 256).astype(np.float32)}
    if with_z:
        input_dict["z"] = np.random.randn(1, 8, 8192).astype(np.float32)

    yield input_dict


def tts_dynamic_axes(with_z=True):
    d = {"spec": {0: "batchsize", 2: "seqlen"}}
    if with_z:
        d["z"] = {0: "batchsize", 2: "zlen"}
    return d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="tts_hifigan.nav",
    )
    return parser.parse_args()


def model_hifigan_qat():
    return TTSWrapper(HifiGanModel.from_pretrained(model_name="tts_hifigan"), with_z=False)


def main():
    args = parse_args()

    dataloader = [next(hifigan_qat_dataloader(with_z=False))]

    package_explicit_int8 = nav.torch.optimize(
        model=model_hifigan_qat(),
        dataloader=dataloader,
        custom_configs=[
            nav.TensorRTConfig(precision=nav.TensorRTPrecision.INT8),
            nav.OnnxConfig(opset=17, dynamic_axes=tts_dynamic_axes(with_z=False)),
        ],
    )

    nav.package.save(package=package_explicit_int8, path=args.output_path)


if __name__ == "__main__":
    main()
