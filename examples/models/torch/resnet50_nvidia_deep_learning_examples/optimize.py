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

import model_navigator as nav


def get_model():
    from image_classification import models as convnet_models  # pytype: disable=import-error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convnet_models.resnet50(pretrained=True).to(device).eval()

    return model


def get_dataloader(data_path: str, batch_size: int):
    from image_classification.dataloaders import get_pytorch_val_loader  # pytype: disable=import-error

    val_dataloader, _ = get_pytorch_val_loader(data_path, 160, batch_size, 10, False)

    return val_dataloader


def get_verification_function(dataloader):
    def verify_top1_accuracy(preds, _):
        correct, total = 0, 0
        for (_, y), output in zip(iter(dataloader), preds):
            y_pred = np.argmax(output["logits"], axis=1)
            total += len(y)
            correct += np.sum(y_pred == y)
        return correct / total

    return verify_top1_accuracy


def get_configuration():
    return {
        "input_names": ("image",),
        "output_names": ("logits",),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="resnet50.nav",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-top1-accuracy",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="imagenette2-160/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = get_model()
    dataloader = get_dataloader(args.data_path, args.batch_size)
    verify_func = get_verification_function(dataloader)
    configuration = get_configuration()

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
        **configuration,
    )

    nav.package.save(package, args.output_path, override=True)


if __name__ == "__main__":
    main()
