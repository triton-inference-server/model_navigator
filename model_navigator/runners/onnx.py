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
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import model_navigator.utils.common as utils
from model_navigator.api.config import Format, TensorType
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata, get_tensor_type
from model_navigator.exceptions import ModelNavigatorNotFoundError
from model_navigator.frameworks.onnx.utils import ONNX_RT_TYPE_TO_NP
from model_navigator.frameworks.tensorrt.cuda import DeviceView
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import module

onnxrt = module.lazy_import("onnxruntime")
np = module.lazy_import("numpy")


class SessionFromOnnx:
    """ONNX session wrapper.

    Functor that builds an ONNX-Runtime inference session.
    """

    def __init__(self, model_bytes: Union[bytes, str], providers: Optional[Sequence[str]] = None):
        """Builds an ONNX-Runtime inference session.

        Args:
            model_bytes: A serialized ONNX model or a path to a model or a callable that returns one of those.

            providers: A sequence of execution providers to use in order of priority.
                    Each element of the sequence may be either an exact match or a case-insensitive partial match
                    for the execution providers available in ONNX-Runtime. For example, a value of "cpu" would
                    match the "CPUExecutionProvider".
                    Defaults to ``["CUDA"]``.

        """
        self._model_bytes_or_path = model_bytes
        self.providers = utils.default(providers, ["cuda"])

    def __call__(self, *args, **kwargs):
        """Invokes ``call_impl``.

        NOTE: ``call_impl`` should *not* be called directly - use this function instead.
        """
        return self.call_impl(*args, **kwargs)

    def call_impl(self):
        """Implementation of ``__call__``.

        Creates an ONNX-Runtime inference session.

        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = utils.invoke_if_callable(self._model_bytes_or_path)

        available_providers = onnxrt.get_available_providers()
        providers = []
        for prov in self.providers:
            matched_prov = utils.find_str_in_iterable(prov, available_providers)
            if matched_prov is None:
                raise ModelNavigatorNotFoundError(
                    f"Could not find specified ONNX-Runtime execution provider.\nNote: Requested provider was: {prov}, but available providers are: {available_providers}"
                )
            providers.append(matched_prov)

        LOGGER.info(f"Creating ONNX-Runtime Inference Session with providers: {providers}")
        return onnxrt.InferenceSession(model_bytes, providers=providers)


class _BaseOnnxrtRunner(NavigatorRunner):
    _provider: str

    def __init__(self, disable_fallback=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._disable_fallback = disable_fallback
        self._sess = SessionFromOnnx(self._model.as_posix(), providers=[self._provider])

    @classmethod
    def format(cls) -> Format:
        return Format.ONNX

    def get_input_metadata_impl(self):
        meta = TensorMetadata()
        for node in self.sess.get_inputs():
            dtype = ONNX_RT_TYPE_TO_NP[node.type] if node.type in ONNX_RT_TYPE_TO_NP else None
            meta.add(node.name, dtype=dtype, shape=node.shape)
        return meta

    def get_onnx_input_metadata(self):
        assert self.is_active and hasattr(self, "sess"), "Runner must be activated."

        input_metadata = TensorMetadata()
        for node in self.sess.get_inputs():
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
        self.sess, _ = utils.invoke_if_callable(self._sess)
        if self._disable_fallback:
            LOGGER.info("Disable fallback for ONNX execution provider.")
            active_providers = self.sess.get_providers()
            if self._provider not in active_providers:
                raise RuntimeError(f"Unable to initialize defined provider: {self._provider}.")

    def deactivate_impl(self):
        del self.sess

    def get_available_input_types(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def get_available_return_types_impl(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]


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

    def infer_impl(self, feed_dict, *args, **kwargs):
        """Run inference."""
        assert self.is_active and hasattr(self, "sess"), "Runner must be activated."

        input_metadata = self.get_onnx_input_metadata()
        feed_dict = {name: self._to_numpy(tensor) for name, tensor in feed_dict.items() if name in input_metadata}

        inference_outputs = self.sess.run(None, feed_dict)
        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), inference_outputs):
            out_dict[node.name] = out

        if self.output_metadata is None:
            return out_dict

        if self.output_metadata:  # filter outputs if output_metadata is set
            out_dict = {k: v for k, v in out_dict.items() if k in self.output_metadata}

        return out_dict

    @staticmethod
    def _to_numpy(tensor):
        tensor_type = get_tensor_type(tensor)
        if tensor_type == TensorType.NUMPY:
            return tensor
        elif tensor_type == TensorType.TORCH:
            return tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")


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

    def infer_impl(self, feed_dict, *args, **kwargs):
        """Run inference."""
        assert self.is_active and hasattr(self, "sess"), "Runner must be activated."

        input_metadata = self.get_onnx_input_metadata()
        feed_dict = {name: tensor for name, tensor in feed_dict.items() if name in input_metadata}
        io_binding = self._get_io_bindings(feed_dict)
        self.sess.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), io_binding.get_outputs()):
            device_view = DeviceView(out.data_ptr(), out.shape(), ONNX_RT_TYPE_TO_NP[out.data_type()])
            out_dict[node.name] = device_view.torch() if self.return_type == TensorType.TORCH else device_view.numpy()

        if self.output_metadata is None:
            return out_dict

        out_dict = {k: v for k, v in out_dict.items() if k in self.output_metadata}
        return out_dict

    def _get_io_bindings(self, feed_dict):
        assert self.is_active and hasattr(self, "sess"), "Runner must be activated."

        io_binding = self.sess.io_binding()
        for name, tensor in feed_dict.items():
            tensor_type = get_tensor_type(tensor)
            if tensor_type == TensorType.TORCH:
                tensor = tensor.to(DeviceKind.CUDA.value)
                io_binding.bind_input(
                    name,
                    DeviceKind.CUDA.value,
                    0,
                    utils.torch_to_numpy_dtype(tensor.dtype),
                    tensor.shape,
                    tensor.data_ptr(),
                )
            else:
                assert tensor_type == TensorType.NUMPY
                io_binding.bind_cpu_input(name, tensor)
        io_binding.synchronize_inputs()

        for name in self.output_metadata:
            io_binding.bind_output(name, DeviceKind.CUDA.value, 0)

        return io_binding


class OnnxrtTensorRTRunner(OnnxrtCUDARunner):
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
