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
from typing import Generator

from model_navigator.converter import ConversionConfig
from model_navigator.converter.config import TargetFormatConfigSetIterator, TensorRTPrecisionMode
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.model import Format


class TensorRTConfigSetIterator(TargetFormatConfigSetIterator):
    def __iter__(self):
        for onnx_opset in self._conversion_set_config.onnx_opsets:
            for target_precision, target_precision_mode in self._precision_modes():
                yield ConversionConfig(
                    target_format=Format.TENSORRT,
                    onnx_opset=onnx_opset,
                    target_precision=target_precision,
                    max_workspace_size=self._conversion_set_config.max_workspace_size,
                    target_precision_mode=target_precision_mode,
                    target_precision_explicit=self._conversion_set_config.target_precisions_mode,
                )

    def _precision_modes(self) -> Generator:
        """
        Generate all possible precision modes based on provided strategy
        """
        target_precisions = self._conversion_set_config.target_precisions
        target_precisions_mode = self._conversion_set_config.target_precisions_mode

        if target_precisions_mode == TensorRTPrecisionMode.HIERARCHY:
            for target_precision in target_precisions:
                yield target_precision, target_precisions_mode
        elif target_precisions_mode == TensorRTPrecisionMode.SINGLE:
            for target_precision in target_precisions:
                yield target_precision, target_precisions_mode
        elif target_precisions_mode == TensorRTPrecisionMode.MIXED:
            for precision_mode in [TensorRTPrecisionMode.HIERARCHY, TensorRTPrecisionMode.SINGLE]:
                for target_precision in target_precisions:
                    yield target_precision, precision_mode
        else:
            raise ModelNavigatorException(f"Unsupported TensorRT target precision mode: {target_precisions_mode}")
