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

import model_navigator as nav

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


dataloader = [torch.randn(1) for _ in range(10)]


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_pyt_export_torchscript():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TORCHSCRIPT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.ONNX,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_torch_trt_script():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TORCH_TRT,),
            jit_options=(nav.JitType.SCRIPT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Output formats
        from packaging import version

        if version.parse(torch.__version__) >= version.parse("1.10"):
            assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_trt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TENSORRT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT)
        assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX) is False


def test_pyt_export_multi_input():
    class MultiInputModule(torch.nn.Module):
        def forward(self, x, y):
            return x + 10

    multi_input_model = MultiInputModule()
    dict_dataloader = [{"x": torch.randn(1), "y": torch.randn(2)} for _ in range(10)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.torch.export(
            model=multi_input_model,
            dataloader=dict_dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "torchscript-script", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=package_dir / "torchscript-trace", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=package_dir / "torch-trt-script", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX)
        assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT)
        assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
