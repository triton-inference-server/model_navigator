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

from pathlib import Path
from typing import Optional, Tuple

import torch  # pytype: disable=import-error

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.export import exporters
from model_navigator.framework_api.commands.export.base import ExportBase
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.execution_context import ExecutionContext
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import JitType, parse_kwargs_to_cmd
from model_navigator.model import Format


class ExportPYT2TorchScript(ExportBase):
    def __init__(self, target_jit_type: JitType, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Export PyTorch to TorchScript",
            command_type=CommandType.EXPORT,
            target_format=Format.TORCHSCRIPT,
            requires=requires,
        )
        self.target_jit_type = target_jit_type

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        target_device: str,
        verbose: bool,
        model: Optional[torch.nn.Module] = None,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("TorchScrip export started")

        exported_model_path = workdir / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            LOGGER.info("Model already exists. Skipping export.")
            return self.get_output_relative_path()

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        model.to(target_device)

        exporters.pytorch2torchscript.get_model = lambda: model

        with ExecutionContext(
            workdir=workdir,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
        ) as context:

            kwargs = {
                "exported_model_path": exported_model_path.relative_to(workdir).as_posix(),
                "target_jit_type": self.target_jit_type.value,
                "batch_dim": batch_dim,
                "target_device": target_device,
                "navigator_workdir": workdir.as_posix(),
            }

            args = parse_kwargs_to_cmd(kwargs, (list, dict, tuple))

            context.execute_local_runtime_script(
                exporters.pytorch2torchscript.__file__, exporters.pytorch2torchscript.export, args
            )

        return self.get_output_relative_path()


class ExportPYT2ONNX(ExportBase):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Export PyTorch to ONNX", command_type=CommandType.EXPORT, target_format=Format.ONNX, requires=requires
        )

    def __call__(
        self,
        workdir: Path,
        model_name: str,
        opset: int,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: str,
        verbose: bool,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
        model: Optional[torch.nn.Module] = None,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> Optional[Path]:
        LOGGER.info("PyTorch to ONNX export started")
        exported_model_path = workdir / self.get_output_relative_path()
        if exported_model_path.exists():
            LOGGER.info("Model already exists. Skipping export.")
            return self.get_output_relative_path()

        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if model is None:
            raise RuntimeError("Expected model of type torch.nn.Module. Got None instead.")

        model.to(target_device)

        exporters.pytorch2onnx.get_model = lambda: model

        with ExecutionContext(
            workdir=workdir,
            script_path=exported_model_path.parent / "reproduce_export.py",
            cmd_path=exported_model_path.parent / "reproduce_export.sh",
            verbose=verbose,
        ) as context:

            kwargs = {
                "navigator_workdir": workdir.as_posix(),
                "exported_model_path": exported_model_path.relative_to(workdir).as_posix(),
                "opset": opset,
                "input_names": list(input_metadata.keys()),
                "output_names": list(output_metadata.keys()),
                "dynamic_axes": dict(**input_metadata.dynamic_axes, **output_metadata.dynamic_axes),
                "batch_dim": batch_dim,
                "forward_kw_names": list(forward_kw_names) if forward_kw_names else None,
                "target_device": target_device,
            }

            args = parse_kwargs_to_cmd(kwargs)

            context.execute_local_runtime_script(exporters.pytorch2onnx.__file__, exporters.pytorch2onnx.export, args)

        return self.get_output_relative_path()
