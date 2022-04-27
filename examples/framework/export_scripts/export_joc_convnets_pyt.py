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
import subprocess
import sys
import tempfile

import torch  # pytype: disable=import-error

import model_navigator as nav

DATALOADER = [torch.randn(2, 3, 224, 224)]

MODEL_NAMES = [
    "resnet50",
    "resnext101-32x4d",
    "se-resnext101-32x4d",
    "efficientnet-b0",
    "efficientnet-b4",
    "efficientnet-widese-b0",
    "efficientnet-widese-b4",
]


def setup_env(workdir):
    cmd = ["git", "clone", "https://github.com/NVIDIA/DeepLearningExamples", f"{workdir}/DeepLearningExamples"]
    subprocess.run(cmd, check=True)
    sys.path.append(f"{workdir}/DeepLearningExamples/PyTorch/Classification/ConvNets/")


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=MODEL_NAMES)
    group.add_argument(
        "--list-models",
        action="store_true",
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    if args.list_models:
        print(MODEL_NAMES)
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_env(tmpdir)
        from image_classification import models as convnet_models  # pytype: disable=import-error

        models = {
            "resnet50": convnet_models.resnet50,
            "resnext101-32x4d": convnet_models.resnext101_32x4d,
            "se-resnext101-32x4d": convnet_models.se_resnext101_32x4d,
            "efficientnet-b0": convnet_models.efficientnet_b0,
            "efficientnet-b4": convnet_models.efficientnet_b4,
            "efficientnet-widese-b0": convnet_models.efficientnet_widese_b0,
            "efficientnet-widese-b4": convnet_models.efficientnet_widese_b4,
        }
        assert set(models.keys()) == set(MODEL_NAMES)
        model = models[args.model_name](pretrained=True).eval()
        pkg_desc = nav.torch.export(
            model=model,
            model_name=f"{args.model_name}_pyt",
            dataloader=DATALOADER,
            override_workdir=True,
        )
        output_path = args.output_path or f"{args.model_name}_pyt.nav"
        pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
