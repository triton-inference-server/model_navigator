# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import gc
from copy import deepcopy
from typing import List, Optional

from model_navigator.api.config import Format, TensorType
from model_navigator.configuration.validation.device import get_id_from_device_string, validate_device_string
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import get_tensor_type
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.frameworks import is_torch2_available
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.runners.base import DeviceKind, InferenceStep, InferenceStepTimer, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import module
from model_navigator.utils.common import numpy_to_torch_dtype

torch = module.lazy_import("torch")


class _BaseTorchRunner(NavigatorRunner):
    """Base runner for inference using PyTorch."""

    _target_device = None

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TORCH

    def __init__(
        self, inference_mode: bool = True, autocast: bool = False, device: Optional[str] = None, *args, **kwargs
    ) -> None:
        """Initialization implementation."""
        super().__init__(*args, **kwargs)
        self._inference_mode = inference_mode
        self._autocast = autocast
        self._loaded_model = None
        self.device = device

        if is_torch2_available():
            self._infer = self._infer_v2
            self._inplace_infer = self._inplace_infer_v2
        else:
            self._infer = self._infer_v1
            self._inplace_infer = self._inplace_infer_v1

        # validate device with runner target device
        if self.device:
            validate_device_string(self.device)
            if self._target_device.value not in self.device:
                raise ModelNavigatorConfigurationError(f"Device type {self.device} is not supported by {self.name()}.")

            device_id = get_id_from_device_string(self.device)
            if device_id and device_id >= torch.cuda.device_count():
                raise ModelNavigatorConfigurationError(
                    f"Device index {device_id} exceeds the number of available devices {torch.cuda.device_count()}."
                )
        else:
            self.device = self._target_device.value

        self._inference_step_timer = InferenceStepTimer(
            self._inference_time, enabled=self._enable_timer, callbacks=[lambda: torch.cuda.synchronize()]
        )

    def activate_impl(self):
        """Activation implementation."""
        self._loaded_model = self.model
        self._loaded_model.to(self.device).eval()

    def deactivate_impl(self):
        """Deactivation implementation."""
        self._loaded_model = None

    def infer_impl(self, feed_dict, *args, **kwargs):
        """Inference handler implementation."""
        outputs = self._infer(feed_dict=feed_dict)

        if self.output_metadata is None:
            return outputs

        with self._inference_step_timer.measure_step(InferenceStep.POSTPROCESSING):
            out_dict = self.output_metadata.flatten_sample(outputs)

        with self._inference_step_timer.measure_step(InferenceStep.D2H_MEMCPY):
            out_dict = self._prepare_outputs(out_dict)

        return out_dict

    def inplace_infer(self, *args, **kwargs):
        """Inplace inference handler implementation."""
        return self._loaded_model(*args, **kwargs)

    def get_available_input_types(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def get_available_return_types_impl(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def _infer_v2(self, feed_dict):
        with torch.inference_mode(mode=self._inference_mode):
            args, kwargs = self._prepare_inputs(feed_dict)

            with self._inference_step_timer.measure_step(InferenceStep.COMPUTE), torch.autocast(
                device_type=self.device, enabled=self._autocast
            ):
                outputs = self._loaded_model(*args, **kwargs)
        return outputs

    def _infer_v1(self, feed_dict):
        with torch.no_grad():
            args, kwargs = self._prepare_inputs(feed_dict)

            with self._inference_step_timer.measure_step(InferenceStep.COMPUTE), torch.autocast(
                device_type=self.device, enabled=self._autocast
            ):
                outputs = self._loaded_model(*args, **kwargs)

        return outputs

    def _inplace_infer_v2(self, *args, **kwargs):
        with torch.inference_mode(mode=self._inference_mode):
            with torch.autocast(device_type=self.device, enabled=self._autocast):
                outputs = self._loaded_model(*args, **kwargs)
        return outputs

    def _inplace_infer_v1(self, *args, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type=self.device, enabled=self._autocast):
                outputs = self._loaded_model(*args, **kwargs)

        return outputs

    def _prepare_inputs(self, feed_dict):
        """Prepare inputs for inference."""
        with self._inference_step_timer.measure_step(InferenceStep.PREPROCESSING):
            inputs = {}
            for input_name, spec in self.input_metadata.items():
                value = feed_dict[input_name]
                if spec.dtype != torch.bfloat16:
                    value = self._to_torch_tensor(value, spec.dtype)
                else:
                    value = torch.from_numpy(value).to(torch.bfloat16)
                inputs[input_name] = value

        with self._inference_step_timer.measure_step(InferenceStep.H2D_MEMCPY):
            for input_name, value in inputs.items():
                inputs[input_name] = value.to(self.device)

        with self._inference_step_timer.measure_step(InferenceStep.PREPROCESSING):
            unflatten_inputs = self.input_metadata.unflatten_sample(inputs, wrap_input=True)
            if isinstance(unflatten_inputs[-1], dict):
                args, kwargs = unflatten_inputs[:-1], unflatten_inputs[-1]
            else:
                args, kwargs = unflatten_inputs, {}
        return args, kwargs

    def _prepare_outputs(self, out_dict):
        """Prepare outputs for inference."""
        for name, outputs in out_dict.items():
            if self.return_type == TensorType.NUMPY:
                out_dict[name] = (
                    # TODO: remove to(torch.float32) once torch.bfloat16 is supported
                    outputs.cpu().detach().numpy()
                    if outputs.dtype != torch.bfloat16
                    else outputs.to(torch.float32).cpu().detach().numpy()
                )
        return out_dict

    def _to_torch_tensor(self, value, dtype):
        tensor_type = get_tensor_type(value)
        if tensor_type == TensorType.TORCH:
            value = value.to(numpy_to_torch_dtype(dtype))
        elif tensor_type == TensorType.NUMPY:
            value = value.astype(dtype)
            value = torch.from_numpy(value)
        else:
            raise ValueError(f"Unsupported type {type(value)}")
        return value


class _BaseTorchScriptRunner(_BaseTorchRunner):
    """Base runner for inference of TorchScript models."""

    @classmethod
    def format(cls) -> Format:
        return Format.TORCHSCRIPT

    def activate_impl(self):
        """Activation implementation."""
        self._loaded_model = torch.jit.load(str(self._model), map_location=self.device).eval()


class TorchCUDARunner(_BaseTorchRunner):
    """Torch model CUDA based runner."""

    _target_device = DeviceKind.CUDA

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCUDA"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        # offload the model from the gpu so other processes can use the memory
        self.model.to("cpu")
        torch.cuda.empty_cache()


class TorchCPURunner(_BaseTorchRunner):
    """Torch model CPU based runner."""

    _target_device = DeviceKind.CPU

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCPU"


class TorchScriptCUDARunner(_BaseTorchScriptRunner):
    """TorchScript model GPU based runner."""

    _target_device = DeviceKind.CUDA

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchScriptCUDA"


class TorchScriptCPURunner(_BaseTorchScriptRunner):
    """TorchScript model CPU based runner."""

    _target_device = DeviceKind.CPU

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchScriptCPU"


class TorchTensorRTRunner(_BaseTorchScriptRunner):
    """TorchScript-TensorRT model runner."""

    _target_device = DeviceKind.CUDA

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TORCH_TRT

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchTensorRT"

    def init_impl(self):
        """Initialization implementation."""
        import torch_tensorrt  # pytype: disable=import-error # noqa: F401

    def _to_torch_tensor(self, value, dtype):
        value = super()._to_torch_tensor(value, dtype)
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

    _target_device = DeviceKind.CUDA

    def __init__(self, *args, **kwargs) -> None:
        """Initialization implementation."""
        # FIXME:
        #  Remove this constructor once inference_mode is fixed for TorchCompile.
        #  Related closed issue: https://github.com/pytorch/pytorch/issues/101151
        super().__init__(*args, **kwargs)
        self._infer = self._infer_v1
        self.fullgraph = False
        self.dynamic = None
        self.backend = "inductor"
        self.mode = None
        self.options = None
        self.disable = False

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCompileCUDA"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    def activate_impl(self):
        """Runner activation implementation."""
        super().activate_impl()
        model_copy = deepcopy(self._loaded_model)
        # offload original model from the gpu so other processes can use the memory
        self.model.to("cpu")
        model_copy.to(self.device).eval()
        LOGGER.info(
            f"Using torch.compile with config: fullgraph={self.fullgraph}, dynamic={self.dynamic}, backend={self.backend}, mode={self.mode}, options={self.options}"
        )
        self._loaded_model = torch.compile(
            model=model_copy,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            backend=self.backend,
            mode=self.mode,
            options=self.options,
        )

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        torch._dynamo.reset()
        # offload the model from the gpu so other processes can use the memory
        self.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()


class TorchCompileCPURunner(_BaseTorchRunner):
    """Torch Compile model CPU based runner."""

    _target_device = DeviceKind.CPU

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchCompileCPU"

    def activate_impl(self):
        """Runner activation implementation."""
        super().activate_impl()
        model_copy = deepcopy(self._loaded_model)
        self._loaded_model = torch.compile(model_copy)

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        torch._dynamo.reset()
        gc.collect()


class _BaseTorchExportedProgramRunner(_BaseTorchRunner):
    """Base runner for inference of Torch ExportedProgram models."""

    @classmethod
    def format(cls) -> Format:
        return Format.TORCH_EXPORTEDPROGRAM

    def activate_impl(self):
        """Activation implementation."""
        self._loaded_model = torch.load(str(self._model), map_location=self.device)


class TorchExportedProgramCPURunner(_BaseTorchExportedProgramRunner):
    """Torch ExportedProgram model CPU based runner."""

    _target_device = DeviceKind.CPU

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchExportedProgramCPU"


class TorchExportedProgramCUDARunner(_BaseTorchExportedProgramRunner):
    """Torch ExportedProgram model GPU based runner."""

    _target_device = DeviceKind.CUDA

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [cls._target_device]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TorchExportedProgramCUDA"


def register_torch2_runners():
    """Register runners in global registry."""
    register_runner(TorchCompileCPURunner)
    register_runner(TorchCompileCUDARunner)
    register_runner(TorchExportedProgramCPURunner)
    register_runner(TorchExportedProgramCUDARunner)
