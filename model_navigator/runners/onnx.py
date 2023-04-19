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
"""ONNX runners."""
from typing import List

from polygraphy.backend.onnxrt import OnnxrtRunner as PolygraphyOnnxrtRunner
from polygraphy.backend.onnxrt import SessionFromOnnx

from model_navigator.api.config import Format
from model_navigator.logger import LOGGER
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils.onnx import ONNX_RT_TYPE_TO_NP
from model_navigator.utils.tensor import TensorMetadata


class _BaseOnnxrtRunner(NavigatorRunner):
    _provider: str

    def __init__(self, disable_fallback=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._disable_fallback = disable_fallback
        self._runner = PolygraphyOnnxrtRunner(
            sess=SessionFromOnnx(self._model.as_posix(), providers=[self._provider]), name=self.name()
        )

    @classmethod
    def format(cls) -> Format:
        return Format.ONNX

    def get_onnx_input_metadata(self):
        assert self.is_active, "Runner must be activated."

        input_metadata = TensorMetadata()
        for node in self._runner.sess.get_inputs():
            dtype = ONNX_RT_TYPE_TO_NP[node.type] if node.type in ONNX_RT_TYPE_TO_NP else None
            shape = tuple(dim if isinstance(dim, int) else -1 for dim in node.shape)
            input_metadata.add(node.name, shape, dtype)
        return input_metadata

    def check_input_metadata(self):
        assert self.input_metadata is not None, "Set `input_metadata`."
        onnx_input_metadata = self.get_onnx_input_metadata()
        for name in self.input_metadata:
            assert self.input_metadata[name].dtype == onnx_input_metadata[name].dtype
            assert self.input_metadata[name].shape == onnx_input_metadata[name].shape

    def activate_impl(self):
        self._runner.activate_impl()
        if self._disable_fallback:
            LOGGER.info("Disable fallback for ONNX execution provider.")
            active_providers = self._runner.sess.get_providers()
            if self._provider not in active_providers:
                raise RuntimeError(f"Unable to initialize defined provider: {self._provider}.")

    def infer_impl(self, feed_dict):
        input_metadata = self.get_onnx_input_metadata()
        feed_dict = {name: tensor for name, tensor in feed_dict.items() if name in input_metadata}
        out_dict = self._runner.infer_impl(feed_dict)
        return {k: v for k, v in out_dict.items() if k in self.output_metadata}

    def deactivate_impl(self):
        self._runner.deactivate_impl()


class OnnxrtCPURunner(_BaseOnnxrtRunner):
    """ONNX runner for CPU runtime provider."""

    _provider = "CPUExecutionProvider"

    @classmethod
    def name(cls) -> str:
        """Get runner name."""
        return "OnnxCPU"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]


class OnnxrtCUDARunner(_BaseOnnxrtRunner):
    """ONNX runner for CUDA runtime provider."""

    _provider = "CUDAExecutionProvider"

    @classmethod
    def name(cls) -> str:
        """Get runner name."""
        return "OnnxCUDA"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]


class OnnxrtTensorRTRunner(_BaseOnnxrtRunner):
    """ONNX runner for TensorRT runtime provider."""

    _provider = "TensorrtExecutionProvider"

    @classmethod
    def name(cls) -> str:
        """Get runner name."""
        return "OnnxTensorRT"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]


def register_onnx_runners() -> None:
    """Register CPU, CUDA and TensorRT ONNX runners."""
    register_runner(OnnxrtCPURunner)
    register_runner(OnnxrtCUDARunner)
    register_runner(OnnxrtTensorRTRunner)
