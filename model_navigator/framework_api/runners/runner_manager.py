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
from typing import Dict, Optional

from polygraphy.backend.base import BaseRunner

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.common import DataObject, TensorMetadata
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.utils import (
    Format,
    JitType,
    RuntimeProvider,
    format2runtimes,
    format_to_relative_model_path,
    get_package_path,
)


class RunnerManager(DataObject):
    def __init__(self, input_metadata: TensorMetadata, output_metadata: TensorMetadata, target_device: str):
        self.input_metadata = input_metadata
        self.output_metadata = output_metadata
        self.target_device = target_device

    def get_runner(
        self,
        workdir: Path,
        model_name: str,
        format: Format,
        jit_type: Optional[JitType] = None,
        precision: Optional[TensorRTPrecision] = None,
        runtime: Optional[RuntimeProvider] = None,
    ) -> BaseRunner:
        """
        Load exported model for given format, jit_type and precision and return Polygraphy runner for given runtime.

        :return
            Polygraphy BaseRunner object: https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/backend/base/runner.py
        """
        model_path = get_package_path(workdir=workdir, model_name=model_name) / format_to_relative_model_path(
            format=format, jit_type=jit_type, precision=precision
        )
        if model_path.exists():
            return self._load_runner(model_path=model_path, format=format, runtime=runtime)
        else:
            raise ValueError("Runner does not exists.")

    @classmethod
    def from_dict(cls, data_dict: Dict):
        return cls(
            input_metadata=TensorMetadata.from_json(data_dict["input_metadata"]),
            output_metadata=TensorMetadata.from_json(data_dict["output_metadata"]),
            target_device=data_dict["target_device"],
        )

    def _load_runner(self, model_path: Path, format: Format, runtime: Optional[RuntimeProvider] = None):
        model_path = model_path.as_posix()
        LOGGER.debug(f"Loading runner from path: {model_path}")

        if runtime is None:
            runtime = format2runtimes(format)

        if format == Format.ONNX:
            from polygraphy.backend.onnxrt import SessionFromOnnx

            from model_navigator.framework_api.runners.onnx import OnnxrtRunner

            if not isinstance(runtime, (tuple, list)):
                runtime = [runtime]
            return OnnxrtRunner(SessionFromOnnx(model_path, providers=runtime))
        elif format == Format.TENSORRT:
            from polygraphy.backend.common import BytesFromPath
            from polygraphy.backend.trt import EngineFromBytes

            from model_navigator.framework_api.runners.trt import TrtRunner

            return TrtRunner(EngineFromBytes(BytesFromPath(model_path)))
        elif format in (Format.TORCHSCRIPT, Format.TORCH_TRT):
            from model_navigator.framework_api.runners.pyt import PytRunner

            if format == Format.TORCH_TRT:  # make sure that torch_tensorrt is initialized
                import torch_tensorrt  # pytype: disable=import-error # noqa: F401

            return PytRunner(
                model_path,
                input_metadata=self.input_metadata,
                output_names=list(self.output_metadata.keys()),
                target_device=self.target_device,
            )
        elif format in (Format.TF_SAVEDMODEL, Format.TF_TRT):
            from model_navigator.framework_api.runners.tf import TFSavedModelRunner

            return TFSavedModelRunner(
                model_path,
                input_metadata=self.input_metadata,
                output_names=list(self.output_metadata.keys()),
            )
        else:
            raise ValueError(f"Unknown format: {format}")
