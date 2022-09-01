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


def test_pyt_tensor_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [torch.randn(1) for _ in range(5)]

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return 2 * x

        model = MyModule()

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            input_names=("input_0",),
            run_profiling=False,
            target_formats=(nav.Format.TORCHSCRIPT,),
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


def test_pyt_sequence_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [[torch.randn(1), torch.randn(1)] for _ in range(5)]

        class MyModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = MyModule()

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            input_names=("input_0", "input_1"),
            run_profiling=False,
            target_formats=(nav.Format.TORCHSCRIPT,),
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


def test_pyt_dict_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]

        class MyModule(torch.nn.Module):
            def forward(self, x, z):
                return x + z

        model = MyModule()

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            input_names=("input_x", "input_z"),
            run_profiling=False,
            target_formats=(nav.Format.TORCHSCRIPT,),
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


def test_pyt_dict_dataloader_with_kwargs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]

        class MyModule(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if z is not None:
                    return x + z
                raise ValueError

        model = MyModule()

        nav.torch.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            input_names=("input_x", "input_z"),
            jit_options=(nav.JitType.SCRIPT,),
            target_formats=(nav.Format.ONNX,),
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


def test_onnx_sequence_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        model = MyModule()

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

        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.onnx.export(
            model=onnx_model_path,
            dataloader=_numpy_dataloader,
            model_name=model_name,
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


def test_onnx_dict_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        model = MyModule()

        onnx_model_path = Path(tmp_dir) / "model.onnx"

        _torch_dataloader = [torch.randn(1) for _ in range(5)]
        _numpy_dataloader = [{"x": t.cpu().detach().numpy()} for t in _torch_dataloader]
        torch.onnx.export(
            model,
            args=_torch_dataloader[0],
            f=onnx_model_path,
            verbose=False,
            opset_version=13,
        )

        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        nav.onnx.export(
            model=onnx_model_path,
            model_name=model_name,
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
