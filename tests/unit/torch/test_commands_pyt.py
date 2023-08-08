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
# pytype: disable=import-error

import pathlib
import tempfile

import numpy
import onnx
import torch  # pytype: disable=import-error

from model_navigator.api.config import DeviceKind, Format, JitType, TensorType
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.correctness import Correctness
from model_navigator.commands.data_dump.samples import samples_to_npz
from model_navigator.commands.export.torch import ExportTorch2ONNX, ExportTorch2TorchScript
from model_navigator.core.tensor import PyTreeMetadata, TensorMetadata, TensorSpec
from model_navigator.core.workspace import Workspace
from model_navigator.runners.torch import TorchScriptCPURunner, TorchScriptCUDARunner
from model_navigator.utils.devices import is_cuda_available

# pytype: enable=import-error

VALUE_IN_TENSOR = 9.0
OPSET = 11


dataloader = [torch.full((1, 1), VALUE_IN_TENSOR) for _ in range(5)]


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_pyt_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        model_dir = workspace / "torchscript-script"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        model_relative_path = pathlib.Path("torchscript-script") / "model.pt"

        script_module = torch.jit.script(model)
        torch.jit.save(script_module, model_path.as_posix())

        input_data = next(iter(dataloader))
        numpy_input = input_data.numpy()
        numpy_output = model(input_data).detach().cpu().numpy()
        batch_dim = None

        samples_to_npz([{"input__1": numpy_input}], workspace / "model_input" / "correctness", batch_dim=batch_dim)
        samples_to_npz([{"output__1": numpy_output}], workspace / "model_output" / "correctness", batch_dim=batch_dim)

        input_metadata = TensorMetadata(
            {"input__1": TensorSpec("input__1", numpy_input.shape, numpy_input.dtype)},
            pytree_metadata=PyTreeMetadata(("input__1",), TensorType.NUMPY),
        )
        output_metadata = TensorMetadata(
            {"output__1": TensorSpec("output__1", numpy_output.shape, numpy_output.dtype)},
            pytree_metadata=PyTreeMetadata("output__1", TensorType.NUMPY),
        )

        command_output = Correctness().run(
            workspace=Workspace(workspace),
            format=Format.TORCHSCRIPT,
            runner_cls=TorchScriptCUDARunner if is_cuda_available() else TorchScriptCPURunner,
            path=model_relative_path,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            batch_dim=batch_dim,
            verbose=False,
        )
        assert command_output.status == CommandStatus.OK


def test_pyt_export_torchscript_script():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_relative_path = pathlib.Path("torchscript-script") / "model.pt"
        exported_model_path = workspace / model_relative_path
        input_data = next(iter(dataloader))
        numpy_data = input_data.cpu().numpy()
        samples_to_npz([{"input__1": numpy_data}], workspace / "model_input" / "profiling", None)

        command_output = ExportTorch2TorchScript().run(
            model=model,
            workspace=Workspace(workspace),
            path=model_relative_path,
            jit_type=JitType.SCRIPT,
            input_metadata=TensorMetadata(
                {"input__1": TensorSpec("input__1", (1, 1), numpy.dtype("float32"))},
                pytree_metadata=PyTreeMetadata("input__1", TensorType.NUMPY),
            ),
            target_device=DeviceKind.CPU,
            strict=True,
            verbose=False,
            custom_args={},
        )
        assert command_output.status == CommandStatus.OK
        torch.jit.load(exported_model_path.as_posix())


def test_pyt_export_torchscript_trace():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_relative_path = pathlib.Path("torchscript-script") / "model.pt"
        exported_model_path = workspace / model_relative_path
        input_data = next(iter(dataloader))
        numpy_data = input_data.cpu().numpy()
        samples_to_npz([{"input__1": numpy_data}], workspace / "model_input" / "profiling", None)

        command_output = ExportTorch2TorchScript().run(
            model=model,
            workspace=Workspace(workspace),
            path=model_relative_path,
            jit_type=JitType.TRACE,
            input_metadata=TensorMetadata(
                {"input__1": TensorSpec("input__1", (1, 1), numpy.dtype("float32"))},
                pytree_metadata=PyTreeMetadata("input__1", TensorType.NUMPY),
            ),
            target_device=DeviceKind.CPU,
            strict=True,
            verbose=False,
            custom_args={},
        )
        assert command_output.status == CommandStatus.OK
        torch.jit.load(exported_model_path.as_posix())


def test_pyt_export_onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"
        model_relative_path = pathlib.Path("onnx") / "model.onnx"
        exported_model_path = workspace / model_relative_path

        device = DeviceKind.CUDA if torch.cuda.is_available() else DeviceKind.CPU

        dataloader_ = (torch.full((3, 5), VALUE_IN_TENSOR, device=device.value) for _ in range(5))
        model_ = torch.nn.Linear(5, 7).to(device.value).eval()

        input_data = next(iter(dataloader_))
        sample = {"input": input_data.detach().cpu().numpy()}
        samples_to_npz([sample], workspace / "model_input" / "profiling", None)

        command_output = ExportTorch2ONNX().run(
            model=model_,
            workspace=Workspace(workspace),
            path=exported_model_path,
            opset=OPSET,
            input_metadata=TensorMetadata(
                {"input": TensorSpec("input", (-1, 5), numpy.dtype("float32"))},
                pytree_metadata=PyTreeMetadata("input", TensorType.NUMPY),
            ),
            output_metadata=TensorMetadata(
                {"output": TensorSpec("output", (-1, 7), numpy.dtype("float32"))},
                pytree_metadata=PyTreeMetadata("output", TensorType.NUMPY),
            ),
            target_device=device,
            verbose=False,
            custom_args={},
        )
        assert command_output.status == CommandStatus.OK
        onnx.checker.check_model(exported_model_path.as_posix())
