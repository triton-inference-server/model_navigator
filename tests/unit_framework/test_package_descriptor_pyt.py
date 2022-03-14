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

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import CommandResults, CommandType
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.pipeline import PipelineResults
from model_navigator.framework_api.utils import Format, Framework, JitType, RuntimeProvider, Status

# pytype: enable=import-error


dataloader = [torch.randn(1) for _ in range(10)]


class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10


model = MyModule()


def test_pyt_package_descriptor():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "torchscript-script"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        script_module = torch.jit.script(model)
        torch.jit.save(script_module, model_path.as_posix())

        config = Config(
            framework=Framework.PYT,
            model_name=model_name,
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            override_workdir=True,
            keep_workdir=True,
            target_formats=(Format.TORCHSCRIPT,),
            target_jit_type=(JitType.SCRIPT,),
            sample_count=1,
            save_data=False,
            disable_git_info=False,
        )

        cmd_export_result = CommandResults(
            name="Mock export command",
            status=Status.OK,
            command_type=CommandType.EXPORT,
            target_format=Format.TORCHSCRIPT,
            target_jit_type=JitType.SCRIPT,
            target_precision=None,
            runtime_provider=RuntimeProvider.DEFAULT,
            missing_params={},
            output=None,
        )

        cmd_correctness_result = CommandResults(
            name="Mock correctness command",
            status=Status.OK,
            command_type=CommandType.CORRECTNESS,
            target_format=Format.TORCHSCRIPT,
            target_jit_type=JitType.SCRIPT,
            target_precision=None,
            runtime_provider=RuntimeProvider.DEFAULT,
            missing_params={},
            output=None,
        )

        pipeline_results = [
            PipelineResults(
                name="Mock pipeline",
                id="mock-pipeline",
                framework=Framework.PYT,
                commands_results=[cmd_export_result, cmd_correctness_result],
            )
        ]

        package_desc = PackageDescriptor(pipeline_results, config)

        # Check model status and load model
        assert package_desc.get_status(format=Format.TORCHSCRIPT, jit_type=JitType.SCRIPT)
        assert package_desc.get_model(format=Format.TORCHSCRIPT, jit_type=JitType.SCRIPT) is not None

        # These models should be not available:
        assert package_desc.get_status(format=Format.TORCHSCRIPT, jit_type=JitType.TRACE) is False
        assert package_desc.get_model(format=Format.TORCHSCRIPT, jit_type=JitType.TRACE) is None

        for jit_type in (JitType.SCRIPT, JitType.TRACE):
            for precision in (TensorRTPrecision.FP16, TensorRTPrecision.FP32):
                assert package_desc.get_status(format=Format.TORCH_TRT, jit_type=jit_type, precision=precision) is False
                assert package_desc.get_model(format=Format.TORCH_TRT, jit_type=jit_type, precision=precision) is None

        assert package_desc.get_status(format=Format.ONNX) is False
        assert package_desc.get_model(format=Format.ONNX) is None
