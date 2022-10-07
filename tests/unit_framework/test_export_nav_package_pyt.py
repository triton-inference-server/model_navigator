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
from model_navigator.utils.devices import get_gpus

# pytype: enable=import-error

CUDA_AVAILABLE = bool(get_gpus(["all"]))


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


dataloader = [torch.randn(1) for _ in range(5)]


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_pyt_export_torchscript():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TORCHSCRIPT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.ONNX,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_torch_trt_script():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TORCH_TRT,),
            jit_options=(nav.JitType.SCRIPT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
            assert (
                check_model_dir(model_dir=workdir / "torch-trt-script-fp16", format=nav.Format.TORCHSCRIPT)
                is CUDA_AVAILABLE
            )

            assert (
                check_model_dir(model_dir=workdir / "torch-trt-script-fp32", format=nav.Format.TORCHSCRIPT)
                is CUDA_AVAILABLE
            )

        # Intermediate formats
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace-fp16", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace-fp32", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_trt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats=(nav.Format.TENSORRT,),
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is CUDA_AVAILABLE
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is CUDA_AVAILABLE

        # Intermediate formats
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False


def test_pyt_export_multi_input():
    class MultiInputModule(torch.nn.Module):
        def forward(self, x, y):
            return x + 10

    multi_input_model = MultiInputModule()
    dict_dataloader = [{"x": torch.randn(1), "y": torch.randn(2)} for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=multi_input_model,
            dataloader=dict_dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT)
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT)
        assert (
            check_model_dir(model_dir=workdir / "torch-trt-script-fp16", format=nav.Format.TORCHSCRIPT)
            is CUDA_AVAILABLE
        )
        assert (
            check_model_dir(model_dir=workdir / "torch-trt-script-fp32", format=nav.Format.TORCHSCRIPT)
            is CUDA_AVAILABLE
        )
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is CUDA_AVAILABLE
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is CUDA_AVAILABLE

        # TODO: refactor tests
        # assert (
        #    check_model_dir(model_dir=workdir / "torch-trt-trace-fp16", format=nav.Format.TORCHSCRIPT) is CUDA_AVAILABLE
        # )
        # assert (
        #    check_model_dir(model_dir=workdir / "torch-trt-trace-fp32", format=nav.Format.TORCHSCRIPT) is CUDA_AVAILABLE
        # )


def test_pyt_export_string_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            target_formats="onnx",
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is False


def test_pyt_export_onnx2trt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_model_path = Path(tmp_dir) / "model.onnx"

        _torch_dataloader = [torch.randn(1) for _ in range(5)]
        _numpy_dataloader = [t.cpu().detach().numpy() for t in _torch_dataloader]

        torch.onnx.export(
            model,
            args=_torch_dataloader[0],
            f=onnx_model_path,
            verbose=False,
            opset_version=13,
        )

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.onnx.export(
            model=onnx_model_path,
            dataloader=_numpy_dataloader,
            opset=13,
            workdir=workdir,
            run_profiling=False,
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

        # Source format copied to package
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)

        # Output formats
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is CUDA_AVAILABLE
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is CUDA_AVAILABLE

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False


def test_pyt_export_onnx_large():
    class LargeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1024, 1024 * 1024)

        def forward(self, x):
            x = self.fc(x)
            return x

    large_model = LargeModel()
    large_dataloader = [torch.randn(1, 1024)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.torch.export(
            model=large_model,
            dataloader=large_dataloader,
            target_formats="onnx",
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            run_profiling=False,
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
        assert check_model_dir(model_dir=workdir / "onnx", format=nav.Format.ONNX)

        # Formats not exported
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "trt-fp32", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torchscript-trace", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-script", format=nav.Format.TORCHSCRIPT) is False
        assert check_model_dir(model_dir=workdir / "torch-trt-trace", format=nav.Format.TORCHSCRIPT) is False
