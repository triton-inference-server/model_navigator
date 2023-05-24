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
"""Torch runners."""
from collections import OrderedDict
from typing import List, Mapping

from model_navigator.api.config import Format, TensorType
from model_navigator.core.tensor import get_tensor_type
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import module
from model_navigator.utils.common import numpy_to_torch_dtype
from model_navigator.utils.dataloader import get_default_output_names

torch = module.lazy_import("torch")


class _BaseTorchRunner(NavigatorRunner):
    """Base runner for inference using PyTorch."""

    _target_device: str

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TORCH

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """Initialization implementation."""
        self._loaded_model = None

    def activate_impl(self):
        """Activation implementation."""
        self._loaded_model = self.model
        self._loaded_model.to(self._target_device).eval()

    def deactivate_impl(self):
        """Deactivation implementation."""
        self._loaded_model = None

    def infer_impl(self, feed_dict):
        """Inference handler implementation."""
        with torch.no_grad():
            inputs = self._prepare_inputs(feed_dict)
            if self.input_metadata_mapping is None:
                outputs = self._loaded_model(*inputs)
            else:
                inputs_dict = dict(zip(self.input_metadata_mapping, inputs))
                outputs = self._loaded_model(**inputs_dict)

        if torch.is_tensor(outputs):
            outputs = (outputs,)
        if isinstance(outputs, Mapping):
            outputs = outputs.values()

        out_dict = OrderedDict()
        if self.output_metadata:
            output_names = self.output_metadata.keys()
        else:
            output_names = outputs.keys() if isinstance(outputs, Mapping) else get_default_output_names(len(outputs))

        for name, output in zip(output_names, outputs):
            out_dict[name] = output
        out_dict = self._prepare_outputs(out_dict)

        return out_dict

    def get_available_input_types(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def get_available_return_types_impl(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def _to_torch_tensor(self, value, dtype):
        tensor_type = get_tensor_type(value)
        if tensor_type == TensorType.TORCH:
            value = value.to(numpy_to_torch_dtype(dtype))
        elif tensor_type == TensorType.NUMPY:
            value = value.astype(dtype)
            value = torch.from_numpy(value)
        else:
            raise ValueError(f"Unsupported type {type(value)}")
        return value.to(self._target_device)

    def _prepare_inputs(self, feed_dict):
        """Prepare inputs for inference."""
        inputs = []
        for input_name, spec in self.input_metadata.items():
            value = feed_dict[input_name]
            value = self._to_torch_tensor(value, spec.dtype)
            inputs.append(value)
        return inputs

    def _prepare_outputs(self, out_dict):
        """Prepare outputs for inference."""
        for name, outputs in out_dict.items():
            if self.return_type == TensorType.NUMPY:
                out_dict[name] = outputs.cpu().numpy()
        return out_dict


class _BaseTorchScriptRunner(_BaseTorchRunner):
    """Base runner for inference of TorchScript models."""

    @classmethod
    def format(cls) -> Format:
        return Format.TORCHSCRIPT

    def activate_impl(self):
        """Activation implementation."""
        self._loaded_model = torch.jit.load(str(self._model), map_location=self._target_device).eval()


class TorchCUDARunner(_BaseTorchRunner):
    """Torch model CUDA based runner."""

    _target_device = "cuda"

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCUDA"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        # offload the model from the gpu so other processes can use the memory
        self.model.to("cpu")
        torch.cuda.empty_cache()


class TorchCPURunner(_BaseTorchRunner):
    """Torch model CPU based runner."""

    _target_device = "cpu"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCPU"


class TorchScriptCUDARunner(_BaseTorchScriptRunner):
    """TorchScript model GPU based runner."""

    _target_device = "cuda"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchScriptCUDA"


class TorchScriptCPURunner(_BaseTorchScriptRunner):
    """TorchScript model CPU based runner."""

    _target_device = "cpu"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchScriptCPU"


class TorchTensorRTRunner(_BaseTorchScriptRunner):
    """TorchScript-TensorRT model runner."""

    _target_device = "cuda"

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TORCH_TRT

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchTensorRT"

    def init_impl(self):
        """Initialization implementation."""
        import torch_tensorrt  # pytype: disable=import-error # noqa: F401

    def _cast_value(self, value, dtype):
        value = value.astype(dtype)
        value = tensorrt_utils.cast_tensor(value)
        return value


def register_torch_runners():
    """Register runners in global registry."""
    register_runner(TorchCUDARunner)
    register_runner(TorchCPURunner)
    register_runner(TorchScriptCUDARunner)
    register_runner(TorchScriptCPURunner)
    register_runner(TorchTensorRTRunner)


class TorchCompileCUDARunner(_BaseTorchRunner):
    """Torch Compile model CUDA based runner."""

    _target_device = "cuda"

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCompileCUDA"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    def activate_impl(self):
        """Runner activation implementation."""
        super().activate_impl()
        self._loaded_model = torch.compile(self._loaded_model)

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        # offload the model from the gpu so other processes can use the memory
        self.model.to("cpu")
        torch.cuda.empty_cache()


class TorchCompileCPURunner(_BaseTorchRunner):
    """Torch Compile model CPU based runner."""

    _target_device = "cpu"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCompileCPU"

    def activate_impl(self):
        """Runner activation implementation."""
        super().activate_impl()
        self._loaded_model = torch.compile(self._loaded_model)


def register_torch2_runners():
    """Register runners in global registry."""
    register_runner(TorchCompileCPURunner)
    register_runner(TorchCompileCUDARunner)
