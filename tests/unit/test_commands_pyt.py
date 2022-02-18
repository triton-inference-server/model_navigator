# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import numpy
import onnx
import torch

from model_navigator.framework_api.commands.correctness.pyt import CorrectnessPYT2TorchScript
from model_navigator.framework_api.commands.data_dump.samples import DumpInputModelData, DumpOutputModelData
from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX, ExportPYT2TorchScript
from model_navigator.framework_api.utils import Framework, JitType
from model_navigator.model import Format
from model_navigator.tensor import TensorSpec

# pytype: enable=import-error

VALUE_IN_TENSOR = 9.0
OPSET = 11


def dataloader():
    for _ in range(10):
        yield torch.full((1, 1), VALUE_IN_TENSOR)


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_pyt_dump_model_input():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_input_dir = package_dir / "model_input"

        input_data = next(dataloader())

        dump_cmd = DumpInputModelData()

        dump_cmd(
            framework=Framework.PYT,
            workdir=workdir,
            model_name=model_name,
            dataloader=dataloader,
            samples=[input_data],
            sample_count=1,
        )

        for sample in [numpy.load(npz_file) for npz_file in model_input_dir.iterdir() if model_input_dir.is_dir()]:
            for dumped, reference in zip([sample[array_name] for array_name in sample.files], [input_data]):
                assert len(dumped) == len(reference)
                assert numpy.allclose(dumped, reference)


def test_pyt_dump_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "torchscript-script"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = package_dir / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = package_dir / "model_output"

        input_data = next(dataloader()).numpy()
        numpy.savez(model_input_dir / "sample.npz", input_data)

        model_output = model(*input_data)

        dump_cmd = DumpOutputModelData()

        dump_cmd(
            framework=Framework.PYT,
            workdir=workdir,
            model=model,
            model_name=model_name,
            dataloader=dataloader,
            samples=[input_data],
            sample_count=1,
        )

        for sample in [numpy.load(npz_file) for npz_file in model_output_dir.iterdir() if model_output_dir.is_dir()]:
            for dumped, reference in zip([sample[array_name] for array_name in sample.files], [model_output]):
                assert numpy.allclose(dumped, reference)


def test_pyt_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "torchscript-script"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"

        script_module = torch.jit.script(model)
        torch.jit.save(script_module, model_path.as_posix())

        input_data = next(dataloader())
        numpy_data = input_data.cpu().numpy()

        correctness_cmd = CorrectnessPYT2TorchScript(target_format=Format.TORCHSCRIPT, target_jit_type=JitType.SCRIPT)

        correctness_cmd(
            framework=Framework.PYT,
            model=model,
            workdir=workdir,
            model_name=model_name,
            rtol=0.0,
            atol=0.0,
            samples=[input_data],
            input_names=("input__1",),
            input_metadata={"input__1": TensorSpec("input__1", numpy_data.shape, numpy_data.dtype)},
            output_names=("output__1",),
        )


def test_pyt_export_torchscript():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"

        export_cmd = ExportPYT2TorchScript(target_jit_type=JitType.SCRIPT)

        exported_model_path = package_dir / export_cmd(
            model=model, model_name=model_name, workdir=workdir, dataloader=dataloader
        )

        torch.jit.load(exported_model_path.as_posix())


def test_pyt_export_onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _dataloader():
            for _ in range(10):
                yield torch.full((3, 5), VALUE_IN_TENSOR, device=device)

        model_ = torch.nn.Linear(5, 7).to(device).eval()

        input_data = next(_dataloader())

        export_cmd = ExportPYT2ONNX()
        exported_model_path = package_dir / export_cmd(
            model=model_,
            model_name=model_name,
            workdir=workdir,
            opset=OPSET,
            input_names=("input",),
            dynamic_axes={"input": {0: "batch"}},
            samples=[input_data],
        )

        onnx.checker.check_model(exported_model_path.as_posix())
