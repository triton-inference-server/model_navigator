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

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch  # pytype: disable=import-error

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.utils import (
    Framework,
    JitType,
    extract_input_shape,
    format_to_relative_model_path,
    get_package_path,
    get_torch_tensor,
    numpy_to_torch_dtype,
    sample_to_tuple,
)
from model_navigator.model import Format


class ExportPYT2TorchScript(Command):
    def __init__(self, target_jit_type: JitType):
        super().__init__(
            name="Export PyTorch to TorchScript",
            command_type=CommandType.EXPORT,
            target_format=Format.TORCHSCRIPT,
        )
        self.target_jit_type = target_jit_type

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format, jit_type=self.target_jit_type)

    def __call__(self, workdir: Path, model, model_name: str, dataloader: Callable, **kwargs) -> Optional[Path]:

        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.target_jit_type == JitType.SCRIPT:
            script_module = torch.jit.script(model)
        else:
            dummy_input = get_torch_tensor(dataloader)
            script_module = torch.jit.trace(model, sample_to_tuple(dummy_input))

        torch.jit.save(script_module, exported_model_path.as_posix())

        return self.get_output_relative_path()


class ExportPYT2ONNX(Command):
    def __init__(self):
        super().__init__(
            name="Export PyTorch to ONNX",
            command_type=CommandType.EXPORT,
            target_format=Format.ONNX,
        )

    def get_output_relative_path(self) -> Path:
        return format_to_relative_model_path(self.target_format)

    def __call__(
        self,
        workdir: Path,
        model,
        model_name: str,
        opset: int,
        samples: List,
        input_names: Optional[Tuple[str]] = None,
        output_names: Optional[Tuple[str]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        **kwargs,
    ) -> Optional[Path]:
        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = samples[0]
        if isinstance(dummy_input, list):
            dummy_input = tuple(dummy_input)
        elif isinstance(dummy_input, dict):
            dummy_input = (dummy_input,)
        input_names_list = list(input_names) if input_names is not None else input_names
        output_names_list = list(output_names) if output_names is not None else output_names

        torch.onnx.export(
            model,
            args=dummy_input,
            f=exported_model_path,
            verbose=False,
            opset_version=opset,
            input_names=input_names_list,
            output_names=output_names_list,
            dynamic_axes=dynamic_axes,
        )

        return self.get_output_relative_path()


class ExportPYT2TorchTensorRT(Command):
    def __init__(self, target_jit_type: JitType):
        super().__init__(
            name="Export PyTorch to TorchTensorRT",
            command_type=CommandType.EXPORT,
            target_format=Format.TORCH_TRT,
        )
        self.target_jit_type = target_jit_type

    def get_output_relative_path(self):
        return format_to_relative_model_path(self.target_format, jit_type=self.target_jit_type)

    def __call__(
        self, workdir: Path, model, model_name: str, opset: int, dataloader: Callable, samples: List, **kwargs
    ) -> Optional[Path]:
        import torch_tensorrt  # pytype: disable=import-error

        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = samples[0]
        input_tensor_spec = extract_input_shape(dataloader, framework=Framework.PYT)

        input_shapes = [
            torch_tensorrt.Input(shape=tensor_spec.shape, dtype=numpy_to_torch_dtype(tensor_spec.dtype))
            for tensor_spec in input_tensor_spec.values()
        ]

        if self.target_jit_type == JitType.TRACE:
            model = torch.jit.trace(model, dummy_input)

        tr_model_compiled = torch_tensorrt.compile(model, inputs=input_shapes)

        tr_model_compiled.save(exported_model_path.as_posix())

        return self.get_output_relative_path()
