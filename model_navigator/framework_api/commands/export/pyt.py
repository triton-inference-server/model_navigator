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
from typing import Dict, List, Optional, Tuple, Union

import torch  # pytype: disable=import-error

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample, TensorMetadata
from model_navigator.framework_api.utils import (
    JitType,
    format_to_relative_model_path,
    get_package_path,
    numpy_to_torch_dtype,
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

    def __call__(
        self,
        workdir: Path,
        model,
        model_name: str,
        profiling_sample: Sample,
        target_device: str,
        input_metadata: TensorMetadata,
        **kwargs,
    ) -> Optional[Path]:

        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        model.to(target_device)
        if self.target_jit_type == JitType.SCRIPT:
            script_module = torch.jit.script(model)
        else:
            dummy_input = tuple(
                torch.from_numpy(val.astype(spec.dtype)).to(target_device)
                for (val, spec) in zip(profiling_sample.values(), input_metadata.values())
            )
            script_module = torch.jit.trace(model, dummy_input)

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
        profiling_sample: Sample,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        target_device: str,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Optional[Path]:
        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = tuple(
            torch.from_numpy(val.astype(spec.dtype)).to(target_device)
            for (val, spec) in zip(profiling_sample.values(), input_metadata.values())
        )
        if forward_kw_names is not None:
            dummy_input = ({key: val for key, val in zip(forward_kw_names, dummy_input)},)

        model.to(target_device)

        torch.onnx.export(
            model,
            args=dummy_input,
            f=exported_model_path,
            verbose=False,
            opset_version=opset,
            input_names=list(input_metadata.keys()),
            output_names=list(output_metadata.keys()),
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
        self, workdir: Path, model, model_name: str, profiling_sample: Sample, input_metadata: TensorMetadata, **kwargs
    ) -> Optional[Path]:
        import torch_tensorrt  # pytype: disable=import-error

        exported_model_path = get_package_path(workdir, model_name) / self.get_output_relative_path()
        if exported_model_path.is_file() or exported_model_path.is_dir():
            return None
        exported_model_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = profiling_sample

        input_shapes = [
            torch_tensorrt.Input(shape=tensor_spec.shape, dtype=numpy_to_torch_dtype(tensor_spec.dtype))
            for tensor_spec in input_metadata.values()
        ]

        if self.target_jit_type == JitType.TRACE:
            model = torch.jit.trace(model, dummy_input)

        tr_model_compiled = torch_tensorrt.compile(model, inputs=input_shapes)

        tr_model_compiled.save(exported_model_path.as_posix())

        return self.get_output_relative_path()
