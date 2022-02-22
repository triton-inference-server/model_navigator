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
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.framework_api.commands.core import CommandType
from model_navigator.framework_api.commands.correctness.base import CorrectnessBase
from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.runners.pyt import PytRunner
from model_navigator.framework_api.runners.trt import TrtRunner
from model_navigator.framework_api.utils import JitType, format_to_relative_model_path, get_package_path
from model_navigator.model import Format


def get_assert_message(atol: float, rtol: float):
    return f"Current atol = {atol}, rtol = {rtol}, try to adjust tolerance values"


class CorrectnessPYT2TorchScript(CorrectnessBase):
    def __init__(self, target_format: Format, target_jit_type: JitType):
        super().__init__(
            name="Correctness PyTorch to TorchScript",
            command_type=CommandType.CORRECTNESS,
            target_format=target_format,
        )
        self.target_jit_type = target_jit_type

    def _get_runners(
        self,
        model: torch.nn.Module,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model_name: str,
        target_device: str,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Tuple[BaseRunner, BaseRunner]:

        output_names = list(output_metadata.keys())

        pyt_runner = PytRunner(
            model,
            input_metadata=input_metadata,
            output_names=output_names,
            target_device=target_device,
            forward_kw_names=forward_kw_names,
        )

        exported_model_path = get_package_path(workdir, model_name) / format_to_relative_model_path(
            format=self.target_format, jit_type=self.target_jit_type
        )
        ts_runner = PytRunner(
            torch.jit.load(exported_model_path),
            input_metadata=input_metadata,
            output_names=output_names,
            target_device=target_device,
        )

        return pyt_runner, ts_runner


class CorrectnessPYT2ONNX(CorrectnessBase):
    def __init__(self):
        super().__init__(
            name="Correctness PyTorch to ONNX",
            command_type=CommandType.CORRECTNESS,
            target_format=Format.ONNX,
        )

    def _get_runners(
        self,
        model: torch.nn.Module,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model_name: str,
        target_device: str,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Tuple[BaseRunner, BaseRunner]:

        pyt_runner = PytRunner(
            model,
            input_metadata=input_metadata,
            output_names=list(output_metadata.keys()),
            target_device=target_device,
            forward_kw_names=forward_kw_names,
        )

        exported_model_path = get_package_path(workdir, model_name) / ExportPYT2ONNX().get_output_relative_path()
        onnx_runner = OnnxrtRunner(SessionFromOnnx(exported_model_path.as_posix()))

        return pyt_runner, onnx_runner


class CorrectnessPYT2TRT(CorrectnessBase):
    def __init__(self, target_precision: TensorRTPrecision):
        super().__init__(
            name="Correctness PyTorch to TensorRT",
            command_type=CommandType.CORRECTNESS,
            target_format=Format.TENSORRT,
        )
        self.target_precision = target_precision

    def _get_runners(
        self,
        model: torch.nn.Module,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        model_name: str,
        target_device: str,
        forward_kw_names: Optional[Tuple[str]] = None,
        **kwargs,
    ) -> Tuple[BaseRunner, BaseRunner]:

        pyt_runner = PytRunner(
            model,
            input_metadata=input_metadata,
            output_names=list(output_metadata.keys()),
            forward_kw_names=forward_kw_names,
            target_device=target_device,
        )

        converted_model_path = (
            get_package_path(workdir, model_name)
            / ConvertONNX2TRT(target_precision=self.target_precision).get_output_relative_path()
        )
        trt_runner = TrtRunner(EngineFromBytes(BytesFromPath(converted_model_path.as_posix())))

        return pyt_runner, trt_runner
