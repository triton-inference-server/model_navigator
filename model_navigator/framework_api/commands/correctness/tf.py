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

import tensorflow  # pytype: disable=import-error
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnxrt import SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2ONNX
from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.commands.correctness.base import CorrectnessBase
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.runners.tf import TFRunner, TFTRTRunner
from model_navigator.framework_api.runners.trt import TrtRunner
from model_navigator.framework_api.utils import RuntimeProvider, format_to_relative_model_path, get_package_path
from model_navigator.model import Format


class CorrectnessTensorFlow2ONNX(CorrectnessBase):
    def __init__(
        self,
        runtime_provider: RuntimeProvider,
        requires: Tuple[Command, ...] = (),
    ):
        super().__init__(
            name="Correctness TensorFlow to ONNX",
            command_type=CommandType.CORRECTNESS,
            target_format=Format.ONNX,
            requires=requires,
        )
        self.runtime_provider = runtime_provider

    def _get_runner(
        self,
        model_name: str,
        workdir: Path,
        **kwargs,
    ):

        with UserErrorContext():

            exported_model_path = (
                get_package_path(workdir, model_name) / ConvertSavedModel2ONNX().get_output_relative_path()
            )
            onnx_runner = OnnxrtRunner(
                SessionFromOnnx(exported_model_path.as_posix(), providers=[self.runtime_provider.value])
            )

        return onnx_runner


class CorrectnessTensorFlow2TRT(CorrectnessBase):
    def __init__(
        self,
        target_precision: Optional[TensorRTPrecision] = None,
        requires: Tuple[Command, ...] = (),
    ):
        super().__init__(
            name="Correctness TensorFlow to TensorRT",
            command_type=CommandType.CORRECTNESS,
            target_format=Format.TENSORRT,
            requires=requires,
        )
        self.target_precision = target_precision

    def _get_runner(
        self,
        model_name: str,
        workdir: Path,
        **kwargs,
    ):

        with UserErrorContext():

            converted_model_path = (
                get_package_path(workdir, model_name)
                / ConvertONNX2TRT(target_precision=self.target_precision).get_output_relative_path()
            )
            trt_runner = TrtRunner(EngineFromBytes(BytesFromPath(converted_model_path.as_posix())))

        return trt_runner


class CorrectnessSavedModel(CorrectnessBase):
    def __init__(
        self,
        target_format: Format,
        target_precision: Optional[TensorRTPrecision] = None,
        requires: Tuple[Command, ...] = (),
    ):
        super().__init__(
            name="Correctness SavedModel",
            command_type=CommandType.CORRECTNESS,
            target_format=target_format,
            requires=requires,
        )
        self.target_precision = target_precision

    def _get_runner(
        self,
        model_name: str,
        workdir: Path,
        input_metadata: TensorMetadata,
        output_metadata: TensorMetadata,
        **kwargs,
    ):

        output_names = list(output_metadata.keys())

        exported_model_path = get_package_path(workdir, model_name) / format_to_relative_model_path(
            format=self.target_format,
            precision=self.target_precision,
        )

        with UserErrorContext():
            if self.target_format == Format.TF_TRT:
                savedmodel_runner = TFTRTRunner(
                    tensorflow.keras.models.load_model(exported_model_path),
                    input_metadata=input_metadata,
                    output_names=output_names,
                )
            else:
                savedmodel_runner = TFRunner(
                    tensorflow.keras.models.load_model(exported_model_path),
                    input_metadata=input_metadata,
                    output_names=output_names,
                )

        return savedmodel_runner
