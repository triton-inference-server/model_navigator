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
import os
import pathlib
from typing import List, Optional

import model_navigator.core.context as ctx
from model_navigator.configuration import Format, TensorType
from model_navigator.configuration.device import get_id_from_device_string, validate_device_string
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import get_tensor_type
from model_navigator.exceptions import ModelNavigatorConfigurationError
from model_navigator.frameworks import is_torch2_available
from model_navigator.frameworks.tensorrt import utils as tensorrt_utils
from model_navigator.frameworks.torch.utils import get_module_device
from model_navigator.runners.base import DeviceKind, InferenceStep, InferenceStepTimer, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import module
from model_navigator.utils.common import numpy_to_torch_dtype, str_to_torch_dtype

torch = module.lazy_import("torch")


class _BaseTorchRunner(NavigatorRunner):
    """Base runner for inference using PyTorch."""

    _target_device = None
    is_native = True

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TORCH

    def __init__(
        self,
        inference_mode: bool = True,
        autocast: bool = True,
        device: Optional[str] = None,
        autocast_dtype: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialization implementation."""
        super().__init__(*args, **kwargs)
        self._inference_mode = inference_mode
        self._autocast = autocast
        self._loaded_model = None
        self.device = device
        if isinstance(autocast_dtype, str):
            autocast_dtype = str_to_torch_dtype(autocast_dtype)
        self._autocast_dtype = autocast_dtype

        if is_torch2_available():
            self._infer = self._infer_v2
            self._infer_native = self._infer_native_v2
        else:
            self._infer = self._infer_v1
            self._infer_native = self._infer_native_v1

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
        self._input_module_device = None

    def activate_impl(self):
        """Activation implementation."""
        self._input_module_device = get_module_device(self.model) or torch.device("cpu")
        self._loaded_model = self.model
        self._loaded_model.to(self.device).eval()
        self._adjust_autocast()

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

    def infer_native(self, *args, **kwargs):
        """Inplace inference handler implementation."""
        args, kwargs = self._prepare_native_inputs(*args, **kwargs)
        return self._infer_native(*args, **kwargs)

    def get_available_input_types(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def get_available_return_types_impl(self) -> List[TensorType]:
        return [TensorType.NUMPY, TensorType.TORCH]

    def _infer_v2(self, feed_dict):
        with torch.inference_mode(mode=self._inference_mode):
            args, kwargs = self._prepare_inputs(feed_dict)
            with (
                self._inference_step_timer.measure_step(InferenceStep.COMPUTE),
                torch.autocast(device_type=self.device, enabled=self._autocast, dtype=self._autocast_dtype),
            ):
                outputs = self._loaded_model(*args, **kwargs)
        return outputs

    def _infer_v1(self, feed_dict):
        with torch.no_grad():
            args, kwargs = self._prepare_inputs(feed_dict)

            with (
                self._inference_step_timer.measure_step(InferenceStep.COMPUTE),
                torch.autocast(device_type=self.device, enabled=self._autocast, dtype=self._autocast_dtype),
            ):
                outputs = self._loaded_model(*args, **kwargs)

        return outputs

    def _infer_native_v2(self, *args, **kwargs):
        with torch.inference_mode(mode=self._inference_mode):
            with torch.autocast(device_type=self.device, enabled=self._autocast, dtype=self._autocast_dtype):
                outputs = self._loaded_model(*args, **kwargs)
        return outputs

    def _infer_native_v1(self, *args, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type=self.device, enabled=self._autocast, dtype=self._autocast_dtype):
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

    def _prepare_native_inputs(self, *args, **kwargs):
        """Prepare inputs for inplace inference for torch based runners."""
        sample = (*args, kwargs)

        input_sample = {}
        for n, t in self.input_metadata.flatten_sample(sample).items():
            if isinstance(t, torch.Tensor) and t.device != self.device:
                t = t.to(self.device)

            input_sample[n] = t

        unflatten_inputs = self.input_metadata.unflatten_sample(input_sample, wrap_input=True)
        if isinstance(unflatten_inputs[-1], dict):
            device_args, device_kwargs = unflatten_inputs[:-1], unflatten_inputs[-1]
        else:
            device_args, device_kwargs = unflatten_inputs, {}

        return device_args, device_kwargs

    def _to_torch_tensor(self, value, dtype):
        tensor_type = get_tensor_type(value)
        if tensor_type == TensorType.TORCH:
            value = value.to(numpy_to_torch_dtype(dtype))
        elif tensor_type == TensorType.NUMPY:
            value = torch.from_numpy(value)
            value = value.to(numpy_to_torch_dtype(dtype))
        else:
            raise ValueError(f"Unsupported type {type(value)}")
        return value

    def _adjust_autocast(self):
        # TODO: Consider better handling for controlling autocast behavior
        try:
            if hasattr(self._loaded_model, "parameters") and self._autocast_dtype is None:
                param_dtype = next(self._loaded_model.parameters()).dtype
                if param_dtype in [torch.bfloat16, torch.int8, torch.uint8]:
                    self._autocast = False
                    LOGGER.warning(f"Model has {param_dtype} parameters. Disabling autocast.")
        except StopIteration:
            LOGGER.warning("Model has no parameters.")


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
        if not self._inplace:
            self.model.to(self._input_module_device)
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


def register_torch_runners():
    """Register runners in global registry."""
    register_runner(TorchCUDARunner)
    register_runner(TorchCPURunner)
    register_runner(TorchScriptCUDARunner)
    register_runner(TorchScriptCPURunner)


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
        self.set_cache_dir()
        super().activate_impl()
        LOGGER.info(
            f"Using torch.compile with config: fullgraph={self.fullgraph}, dynamic={self.dynamic}, backend={self.backend}, mode={self.mode}, options={self.options}"
        )
        self._loaded_model = torch.compile(
            model=self._loaded_model,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            backend=self.backend,
            mode=self.mode,
            options=self.options,
        )

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        # offload the model from the gpu so other processes can use the memory
        if not self._inplace:
            self.model.to(self._input_module_device)

        if is_torch2_available():
            torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()

    def set_cache_dir(self):
        """For each runner name there is separate cache dir."""
        torch_inductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if torch_inductor_cache_dir:
            LOGGER.debug("TORCHINDUCTOR_CACHE_DIR: {} set by the user.", torch_inductor_cache_dir)
            return

        workspace = ctx.global_context.get(ctx.INPLACE_OPTIMIZE_WORKSPACE_CONTEXT_KEY)
        if workspace:
            workspace = pathlib.Path(workspace)
        else:
            workspace = pathlib.Path("/tmp/model_navigator/.torch_compile_cache")

        workspace = workspace / self.name() / ".torch_compile_cache"
        workspace.mkdir(parents=True, exist_ok=True)

        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(workspace)


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
        self._loaded_model = torch.compile(self._loaded_model)

    def deactivate_impl(self):
        """Deactivation implementation."""
        super().deactivate_impl()
        if is_torch2_available():
            torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()


class _BaseTorchExportedProgramRunner(_BaseTorchRunner):
    """Base runner for inference of Torch ExportedProgram models."""

    @classmethod
    def format(cls) -> Format:
        return Format.TORCH_EXPORTEDPROGRAM

    def activate_impl(self):
        """Activation implementation."""
        exported_program = torch.export.load(str(self._model))
        self._loaded_model = exported_program.module()
        self._loaded_model.to(self.device)
        self._adjust_autocast()


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


class TorchTensorRTRunner(_BaseTorchExportedProgramRunner):
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


def register_torch2_runners():
    """Register runners in global registry."""
    register_runner(TorchCompileCPURunner)
    register_runner(TorchCompileCUDARunner)
    register_runner(TorchExportedProgramCPURunner)
    register_runner(TorchExportedProgramCUDARunner)
    register_runner(TorchTensorRTRunner)
