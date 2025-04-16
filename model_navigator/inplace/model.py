# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Inplace models."""

import abc
import collections
import copy
import gc
import inspect
import pathlib
import tempfile
from typing import Any, Callable, List, Optional

from model_navigator.configuration import (
    Framework,
    OnnxConfig,
    RuntimeSearchStrategy,
    TensorType,
)
from model_navigator.core.dataloader import to_numpy
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import PyTreeMetadata
from model_navigator.frameworks import is_torch2_available
from model_navigator.package import Package, load_from_workspace
from model_navigator.utils.module import lazy_import

from ..core import context as ctx
from ..exceptions import ModelNavigatorRuntimeError, ModelNavigatorUserInputError
from ..utils.format_helpers import is_source_format
from .config import OptimizeConfig, inplace_config
from .utils import TorchDataloader, get_dynamic_axes_from_shapes

torch = lazy_import("torch")

PYTREE_METADATA_PREFIX = "input"


class BaseModule(abc.ABC):
    """Base class for inplace Optimize modules."""

    def __init__(
        self,
        module,
        name: str,
        input_mapping: Callable,
        output_mapping: Callable,
        optimize_config: Optional[OptimizeConfig] = None,
        forward: Optional[Callable] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be optimized.
            name: name of the module.
            input_mapping: function mapping module input to runner input.
            output_mapping: function mapping runner output to module output.
            optimize_config: configuration for module optimization.
            forward: method to replace navigator default __call__
            device: Device on which optimized module has to be executed.
        """
        if not isinstance(module, torch.nn.Module):
            raise ModelNavigatorUserInputError("Only torch modules are supported.")

        self._module = module
        self._name = name
        self._signature = self._get_signature()
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._device = device
        self._forward_call = forward if forward is not None else self._module
        self._recorder = False
        if optimize_config:
            self.optimize_config = self._update_optimize_config(optimize_config)
        else:
            self.optimize_config = optimize_config

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Module forward method."""
        pass

    @abc.abstractmethod
    def deactivate(self):
        """Deactivate module."""
        pass

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @property
    def module(self) -> "torch.nn.Module":
        """Module object."""
        return self._module

    @property
    def forward_call(self) -> Callable:
        """Module object."""
        return self._forward_call

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return False

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return False

    @property
    def packages(self) -> List[Package]:
        """Get list of packages for optimized modules."""
        return []

    @property
    def _workspace(self):
        return pathlib.Path(inplace_config.cache_dir) / f"{self._name}"

    def _update_optimize_config(self, optimize_config: OptimizeConfig) -> OptimizeConfig:
        config = optimize_config.clone()
        if config.workspace is None:
            config.workspace = self._workspace
            LOGGER.info(f"Setting workspace to {config.workspace}")

        return config

    def _get_signature(self) -> List[str]:
        """Get signature of the module forward method."""
        forward_signature = inspect.signature(self._module.forward)
        forward_params = list(forward_signature.parameters.keys())
        return forward_params


class RecordingModule(BaseModule):
    """Module that records samples for optimization."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize RecordingModule."""
        super().__init__(*args, **kwargs)
        self._samples = collections.defaultdict(list)
        self._samples_shapes = collections.defaultdict(list)
        self._optimized = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{self._name}_")
        self._samples_dir = pathlib.Path(self._temp_dir.name)
        self._min_batch_sizes = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module."""
        LOGGER.debug(f"Calling recording `{self.name}` module.")
        if self.optimize_config:
            self.record_sample(*args, **kwargs)

        output = self._forward_call(*args, **kwargs)

        if self.optimize_config:
            self._recorder = True

        return output

    def deactivate(self):
        """Deactivate module."""
        return

    def optimize(self) -> None:
        """Optimize the module using the recorded samples."""
        from model_navigator.torch import optimize

        if not self._recorder:
            raise ModelNavigatorUserInputError(
                f"""The module `{self.name}` has not been evaluated as part of provided callable. """
                """Please, review the wrapped modules. We use `__call__` method to run queries on module. """
                """If you model use other method, use `forward_func` in module configuration to override the default."""
            )

        if not self.optimize_config:
            raise ModelNavigatorRuntimeError(f"The module `{self.name}` has no optimize configuration")

        batch_dim = 0 if self.optimize_config.batching else None
        if self.optimize_config.batching:
            self._update_max_batch_size()

        config_dict = {k: v for k, v in self.optimize_config.to_dict().items() if k != "workspace"}

        for module_graph_id, pytree_metadata in enumerate(self._samples):
            config_dict["workspace"] = self.optimize_config.workspace / str(module_graph_id)
            samples = self._samples[pytree_metadata]
            samples_shapes = self._samples_shapes[pytree_metadata]
            dynamic_axes = get_dynamic_axes_from_shapes(samples_shapes, pytree_metadata, batch_dim)
            updated_config_dict = self._update_custom_configs(config_dict, dynamic_axes)

            with ctx.global_context.temporary() as tmp_ctx:
                tmp_ctx.set(ctx.INPLACE_OPTIMIZE_WORKSPACE_CONTEXT_KEY, config_dict["workspace"])
                tmp_ctx.set(ctx.INPLACE_OPTIMIZE_MODULE_GRAPH_ID_CONTEXT_KEY, str(module_graph_id))

                optimize(model=self._module, dataloader=TorchDataloader(samples), **updated_config_dict)

        self._optimized = True

    def record_sample(self, *args: Any, **kwargs: Any) -> None:
        """Record a sample from the module."""
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)

        if len(self._samples[pytree_metadata]) < inplace_config.max_num_samples_stored:
            ind = self.get_total_num_samples()
            sample_path = self._samples_dir / f"{ind}.pt"
            torch.save(sample, sample_path)
            self._samples[pytree_metadata].append(sample_path)

        shapes = {n: to_numpy(t, Framework.TORCH).shape for n, t in pytree_metadata.flatten_sample(sample).items()}

        if self.optimize_config.batching:
            recording_batch = ctx.global_context.get(ctx.INPLACE_OPTIMIZE_BATCH_CONTEXT_KEY)
            min_shapes = [shape[0] for shape in shapes.values() if len(shape) > 1]
            if len(min_shapes) > 0:
                batch_size = min(min_shapes)
                if recording_batch in self._min_batch_sizes:
                    self._min_batch_sizes[recording_batch] = min(batch_size, self._min_batch_sizes[recording_batch])
                else:
                    self._min_batch_sizes[recording_batch] = batch_size

        self._samples_shapes[pytree_metadata].append(shapes)

    def get_total_num_samples(self) -> int:
        """Get the total number of samples."""
        return sum(len(samples) for samples in self._samples.values())

    def get_total_num_samples_shapes(self) -> int:
        """Get the total number of samples."""
        return sum(len(samples) for samples in self._samples_shapes.values())

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return (
            self.get_total_num_samples_shapes() >= inplace_config.min_num_samples
        )  # TODO min num samples should be per graph or per module?

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return self._optimized

    def _update_custom_configs(self, config_dict, dynamic_axes):
        config_dict = copy.deepcopy(config_dict)

        if config_dict["custom_configs"] is None:
            config_dict["custom_configs"] = []

        onnx_config = None
        for custom_config in config_dict["custom_configs"]:
            if isinstance(custom_config, OnnxConfig):
                onnx_config = custom_config

        if onnx_config is not None and onnx_config.dynamic_axes is None:
            onnx_config.dynamic_axes = dynamic_axes
        else:
            config_dict["custom_configs"].append(OnnxConfig(dynamic_axes=dynamic_axes))

        return config_dict

    def _update_max_batch_size(self):
        if self.optimize_config.optimization_profile is None:
            return

        max_batch_size = self.optimize_config.optimization_profile.max_batch_size
        batch_sizes = self.optimize_config.optimization_profile.batch_sizes
        if max_batch_size is not None or batch_sizes is not None:
            if not self._min_batch_sizes:
                raise ModelNavigatorUserInputError(
                    """Unable to collect required batch size from input samples."""
                    f"""Has the wrapped module `{self.name}` been executed in scope of the provided function?"""
                )

            scale_ratios = [b // rb for rb, b in self._min_batch_sizes.items()]
            scale_ratio = max(scale_ratios)
            if scale_ratio != 1:
                batch_sizes_info = []
                for rb, b in self._min_batch_sizes.items():
                    batch_sizes_info.append(f"""Pipeline batch size: {rb:6}, module batch size: {b:6}""")

                batch_sizes_info_str = "\n".join(batch_sizes_info)

                LOGGER.info(
                    f"""Found `{self.name}` module is executed with different batch size than other modules.\n"""
                    """Collected batch sizes for module during recording:\n"""
                    f"""{batch_sizes_info_str}"""
                )

            if max_batch_size is not None:
                new_max_batch_size = max_batch_size * scale_ratio
                if new_max_batch_size > max_batch_size:
                    self.optimize_config.optimization_profile.max_batch_size = new_max_batch_size
                    LOGGER.info(
                        f"""Setting maximal batch for size to `{new_max_batch_size}` for {self.name} module as provided"""
                        f"""value `{max_batch_size}`is lower than minimal batch size required for module conversion."""
                    )
            else:
                max_batch_size = max(batch_sizes)
                new_max_batch_size = max_batch_size * scale_ratio
                if new_max_batch_size > max_batch_size:
                    self.optimize_config.optimization_profile.batch_sizes.append(new_max_batch_size)
                    LOGGER.info(
                        f"""Extending batch sizes list with `{new_max_batch_size}`for {self.name} module as """
                        f"""max batch size `{max_batch_size}` provided in `batch_sizes` list is lower than minimal  """
                        """required for module conversion."""
                    )


class OptimizedModule(BaseModule):
    """Module that runs the optimized module."""

    def __init__(
        self, *args, strategies: Optional[List[RuntimeSearchStrategy]] = None, activate_runners: bool = True, **kwargs
    ) -> None:
        """Initialize OptimizedModule.

        Load the optimized module from the Model Navigator workspace
        and get the runner according to the configured strategy.
        """
        super().__init__(*args, **kwargs)

        self._packages = []
        self._runners = {}
        strategies = strategies or inplace_config.strategies

        for package_workspace in self._workspace.iterdir():
            package = load_from_workspace(package_workspace)
            package.load_source_model(self._module)

            runner = package.get_runner(
                return_type=TensorType.TORCH, strategies=strategies, device=self._device, inplace=True
            )

            pytree_metadata = package.status.input_metadata.pytree_metadata

            self._packages.append(package)
            self._runners[pytree_metadata] = runner

        # Fix for multi-graph models
        duplicated_runners = {}
        for pytree_metadata, runner in self._runners.items():
            if is_source_format(runner.format()):
                if runner.name() in duplicated_runners:
                    self._runners[pytree_metadata] = duplicated_runners[runner.name()]
                else:
                    duplicated_runners[runner.name()] = runner

        if not all(is_source_format(runner.format()) for runner in self._runners.values()):
            self._offload_module()

        if activate_runners:
            self._activate_runners()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call through the optimized module."""
        LOGGER.debug(f"Calling optimized `{self.name}` module on device `{self._device}`.")
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        # TODO when only one runner is present, we can avoid the PyTreeMetadata
        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)
        if pytree_metadata not in self._runners:
            raise ValueError(f"No runner found for {pytree_metadata}")
        runner = self._runners[pytree_metadata]

        if runner.is_native:
            output = runner.infer_native(*args, **kwargs)
        else:
            input_dict = runner.input_metadata.flatten_sample(sample)
            runner_output = runner.infer(input_dict)
            output = runner.output_metadata.unflatten_sample(runner_output)
        output = self._output_mapping(output)  # TODO better solution?
        return output

    def deactivate(self):
        """Deactivate the optimized module."""
        self._deactivate_runners()

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return True

    @property
    def packages(self) -> List[Package]:
        """Get the list of packages."""
        return self._packages

    def _activate_runners(self):
        """Activate all runners."""
        LOGGER.info("Activating runners for optimized module.")
        for runner in self._runners.values():
            runner.activate()

    def _deactivate_runners(self):
        """Deactivate all runners."""
        for runner in self._runners.values():
            runner.deactivate()

    def _offload_module(self):
        LOGGER.info("Offloading module to CPU.")
        self._module.to("cpu")
        if torch.cuda.is_available():
            # empty torch cuda buffers after moving module to cpu i.e. more free vram
            if is_torch2_available():
                torch._dynamo.reset()
            torch.cuda.empty_cache()
            gc.collect()


class EagerModule(BaseModule):
    """Module that passes through the original module."""

    def __init__(
        self,
        module,
        name: str,
        input_mapping: Callable,
        output_mapping: Callable,
        optimize_config: Optional[OptimizeConfig] = None,
        forward: Optional[Callable] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be optimized.
            name: name of the module.
            input_mapping: function mapping module input to runner input.
            output_mapping: function mapping runner output to module output.
            optimize_config: configuration for module optimization.
            forward: method to replace navigator default __call__
            device: Device on which optimized module has to be executed.
        """
        super().__init__(
            module=module,
            name=name,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            optimize_config=optimize_config,
            forward=forward,
            device=device,
        )
        self._pytree_metadata = None
        self._module.to(device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Native inference on wrapped module."""
        LOGGER.debug(f"Calling eager `{self.name}` module on device `{self._device}`.")
        args, kwargs = self._prepare_inputs(*args, **kwargs)
        return self._forward_call(*args, **kwargs)

    def deactivate(self):
        """Deactivate module."""
        return

    def _collect_pytree_metadata(self, *args, **kwargs):
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)

        return pytree_metadata

    def _prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for inplace inference and place them on the same device if are not."""
        if not self._pytree_metadata:
            self._pytree_metadata = self._collect_pytree_metadata(*args, **kwargs)

        sample = (*args, kwargs)
        sample = self._input_mapping(sample)

        input_sample = {}
        for n, t in self._pytree_metadata.flatten_sample(sample).items():
            if isinstance(t, torch.Tensor) and t.device != self._device:
                t = t.to(self._device)

            input_sample[n] = t

        unflatten_inputs = self._pytree_metadata.unflatten_sample(input_sample, wrap_input=True)
        if isinstance(unflatten_inputs[-1], dict):
            device_args, device_kwargs = unflatten_inputs[:-1], unflatten_inputs[-1]
        else:
            device_args, device_kwargs = unflatten_inputs, {}

        return device_args, device_kwargs
