#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting PyTorch identity model"""

import argparse
import pathlib

import yaml
from loguru import logger

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}

EXPECTED_STATUES = [
    "onnx.OnnxCUDA",
    "onnx.OnnxTensorRT",
    "torch.TorchCUDA",
    "trt-fp16.TensorRT",
]


def main():
    import numpy as np
    import torch  # pytype: disable=import-error

    import model_navigator as nav
    from model_navigator.commands.base import CommandStatus
    from tests.functional.common.utils import collect_optimize_status, validate_status

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        required=True,
        help="Status file where per path result is stored.",
    )

    args = parser.parse_args()

    logger.debug(f"CLI args: {args}")

    model = torch.nn.Identity()
    dataloader = [{"input": torch.randn(2, 3, dtype=torch.half)} for _ in range(2)]

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(np.allclose(a, b) for a, b in zip(y_runner.values(), y_expected.values())):
                return False
        return True

    package = nav.torch.optimize(
        model=model,
        dataloader=dataloader,
        verify_func=verify_func,
        verbose=True,
        optimization_profile=nav.OptimizationProfile(batch_sizes=[1, 8, 16], stability_percentage=100),
        target_formats=(nav.Format.ONNX, nav.Format.TENSORRT),
        runners=(
            "TorchCUDA",
            "OnnxCUDA",
            "OnnxTensorRT",
            "TensorRT",
        ),  # TODO remove after updating custom runners
        custom_configs=[nav.TensorRTConfig(precision=(nav.TensorRTPrecision.FP16))],
        input_names=("input",),
    )

    status_file = args.status
    status = collect_optimize_status(package.status)
    validate_status(status, expected_statuses=EXPECTED_STATUES)

    numpy_input = {k: v.numpy() for k, v in dataloader[0].items()}
    torch_cpu_input = {k: v.to("cpu") for k, v in dataloader[0].items()}
    torch_cuda_input = {k: v.to("cuda") for k, v in dataloader[0].items()}
    for key, runner_status in status.items():
        if runner_status != CommandStatus.OK.name:
            logger.info(f"Skipping {key} due to {runner_status} status")
            continue
        model_name, runner_name = key.split(".")
        runner = package.get_runner(nav.SelectedRuntimeStrategy(model_name, runner_name))

        assert nav.TensorType.TORCH in runner.get_available_input_types()
        assert nav.TensorType.TORCH in runner.get_available_return_types()

        with runner:
            numpy_output_from_numpy = runner.infer(numpy_input)
            numpy_output_from_torch_cpu = runner.infer(torch_cpu_input)
            numpy_output_from_torch_cuda = runner.infer(torch_cuda_input)

        runner._return_type = nav.TensorType.TORCH
        with runner:
            torch_output_from_numpy = runner.infer(numpy_input)
            assert all(t.is_cuda for t in torch_output_from_numpy.values())
            torch_output_from_numpy = {k: v.detach().cpu().numpy() for k, v in torch_output_from_numpy.items()}

            torch_output_from_torch_cpu = runner.infer(torch_cpu_input)
            assert all(t.is_cuda for t in torch_output_from_torch_cpu.values())
            torch_output_from_torch_cpu = {k: v.detach().cpu().numpy() for k, v in torch_output_from_torch_cpu.items()}

            torch_output_from_torch_cuda = runner.infer(torch_cuda_input)
            assert all(t.is_cuda for t in torch_output_from_torch_cuda.values())
            torch_output_from_torch_cuda = {
                k: v.detach().cpu().numpy() for k, v in torch_output_from_torch_cuda.items()
            }

        for comp_output in [
            numpy_output_from_torch_cpu,
            numpy_output_from_torch_cuda,
            torch_output_from_numpy,
            torch_output_from_torch_cpu,
            torch_output_from_torch_cuda,
        ]:
            assert all(np.allclose(a, b) for a, b in zip(numpy_output_from_numpy.values(), comp_output.values()))

        logger.info(f"Verified {runner.name()}")

    with status_file.open("w") as fp:
        yaml.safe_dump(status, fp)

    logger.info(f"Status saved to {status_file}")


if __name__ == "__main__":
    main()
