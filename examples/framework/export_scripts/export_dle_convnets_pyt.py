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

import numpy as np
import torch  # pytype: disable=import-error

import model_navigator as nav

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


def get_top1_accuracy(runner, dataloader):
    correct, total = 0, 0
    with runner:
        for x, y in iter(dataloader):
            x, y = x.cpu().numpy(), y.cpu().numpy()
            output = runner.infer({"image": x})
            y_pred = np.argmax(output["logits"], axis=1)
            total += len(y)
            correct += np.sum(y_pred == y)
    return correct / total


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
    parser.add_argument(
        "--data-path",
        type=str,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-top1-accuracy",
        type=float,
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
        from image_classification.dataloaders import get_pytorch_val_loader

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models[args.model_name](pretrained=True).to(device).eval()

        if not args.data_path:
            raise RuntimeError(
                """No data path specified. Please download and prepera validation dataset following these instrucions:
            https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5#2-download-and-preprocess-the-dataset
            """
            )

        val_dataloader, _ = get_pytorch_val_loader(
            args.data_path, args.image_size, args.batch_size, args.num_classes, False
        )

        pkg_desc = nav.torch.export(
            model=model,
            model_name=f"{args.model_name}_pyt",
            dataloader=val_dataloader,
            override_workdir=True,
            target_device=device,
            input_names=("image",),
            output_names=("logits",),
            onnx_runtimes=nav.RuntimeProvider.CUDA,
        )

        if args.min_top1_accuracy is not None:
            for model_status in pkg_desc.navigator_status.model_status:
                for runtime_results in model_status.runtime_results:
                    if runtime_results.status == nav.Status.OK:
                        runner = pkg_desc.get_runner(
                            format=model_status.format,
                            jit_type=model_status.torch_jit,
                            precision=model_status.precision,
                            runtime=runtime_results.runtime,
                        )
                        acc = get_top1_accuracy(runner, val_dataloader)
                        nav.LOGGER.info(
                            f"For {model_status.format=}, {model_status.torch_jit=}, {model_status.precision=}, {runtime_results.runtime=} accuracy is {acc}."
                        )
                        if acc > args.min_top1_accuracy:
                            pkg_desc.set_verified(
                                format=model_status.format,
                                jit_type=model_status.torch_jit,
                                precision=model_status.precision,
                                runtime=runtime_results.runtime,
                            )
                            nav.LOGGER.info("Verified.")
                        else:
                            nav.LOGGER.warning(f"Not verified. Min accuracy to be verified is {args.min_top1_accuracy}")
        else:
            nav.LOGGER.info("Provide min-top1-accuracy to verify output formats.")

        output_path = args.output_path or f"{args.model_name}_pyt.nav"
        nav.save(pkg_desc, output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
