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
# pytype: disable=import-error
import tempfile
from pathlib import Path

import torch

import model_navigator.framework_api as nav

# pytype: enable=import-error


def check_model_dir(model_dir: Path, format: nav.Format) -> bool:
    if not model_dir.is_dir():
        return False
    if not Path(model_dir / "config.yaml").is_file():
        return False
    if format == nav.Format.ONNX:
        model_path = model_dir / "model.onnx"
    elif format == nav.Format.TENSORRT:
        model_path = model_dir / "model.plan"
    else:
        model_path = model_dir / "model.pt"
    if not model_path.exists():
        return False

    return True


def test_pyt_export_hf_distilbert():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "distilbert-base-uncased"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.huggingface.torch.export(
            model_name=model_name,
            override_workdir=True,
            workdir=workdir,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all([path.suffix == ".json" for path in model_input_dir.iterdir()])
        assert model_output_dir.is_dir()
        assert all([path.suffix == ".json" for path in model_output_dir.iterdir()])
        assert navigator_log_file.is_file()

        # Passed formats
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX)

        # Failed formats
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False

        # Output formats
        from packaging import version

        if version.parse(torch.__version__) > version.parse("1.10.1"):
            assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT)
            assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT)
        else:
            assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
            assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT) is False
