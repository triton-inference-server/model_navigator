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
"""Inplace models."""

import abc
import collections
import copy
import dataclasses
import inspect
import pathlib
import tempfile
from typing import Any, Callable, List, Optional

from model_navigator.api.config import Framework, OnnxConfig, TensorRTConfig, TensorType, TorchTensorRTConfig
from model_navigator.api.package import load_from_workspace
from model_navigator.core.dataloader import to_numpy
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import PyTreeMetadata
from model_navigator.runtime_analyzer.strategy import RuntimeSearchStrategy
from model_navigator.utils.module import lazy_import

from .config import OptimizeConfig, inplace_config
from .registry import module_registry
from .utils import TorchDataloader, get_dynamic_axes_from_shapes, get_trt_profile_from_shapes

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
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be optimized.
            name: name of the module.
            input_mapping: function mapping module input to runner input.
            output_mapping: function mapping runner output to module output.
            optimize_config: configuration for module optimization.
        """
        self._module = module
        self._name = name
        self._signature = self._get_signature()
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        if optimize_config:
            self._optimize_config = self._update_optimize_config(optimize_config)
        else:
            self._optimize_config = optimize_config

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @abc.abstractclassmethod
    def __call__(cls, *args, **kwargs) -> Any:
        """Module forward method."""
        pass

    @property
    def _workspace(self):
        return pathlib.Path(inplace_config.cache_dir) / f"{self._name}"

    def _update_optimize_config(self, optimize_config: OptimizeConfig) -> OptimizeConfig:
        config = dataclasses.replace(optimize_config)
        if config.workspace is None:
            config.workspace = self._workspace
            LOGGER.info(f"Setting workspace to {config.workspace}")

        return config

    def _get_signature(self) -> List[str]:
        """Get signature of the module forward method."""
        return inspect.getfullargspec(self._module.forward).args[1:]

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for optimization."""
        return False

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return False


class RecordModule(BaseModule):
    """Module that records samples for optimization."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize RecordModule."""
        super().__init__(*args, **kwargs)
        self._samples = collections.defaultdict(list)
        self._samples_shapes = collections.defaultdict(list)
        self._optimized = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{self._name}_")
        self._samples_dir = pathlib.Path(self._temp_dir.name)

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
        self._samples_shapes[pytree_metadata].append(shapes)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module."""
        self.record_sample(*args, **kwargs)
        output = self._module(*args, **kwargs)
        return output

    def optimize(self):
        """Optimize the module using the recorded samples."""
        from model_navigator.api.torch import optimize

        batch_dim = 0 if self._optimize_config.batching else None
        if self._optimize_config.optimization_profile is not None:
            max_batch_size = self._optimize_config.optimization_profile.max_batch_size
        else:
            max_batch_size = None
        config_dict = {k: v for k, v in self._optimize_config.to_dict().items() if k != "workspace"}

        for i, pytree_metadata in enumerate(self._samples):
            config_dict["workspace"] = self._optimize_config.workspace / str(i)
            samples = self._samples[pytree_metadata]
            samples_shapes = self._samples_shapes[pytree_metadata]
            dynamic_axes = get_dynamic_axes_from_shapes(samples_shapes, pytree_metadata, batch_dim)
            trt_profile = get_trt_profile_from_shapes(samples_shapes, pytree_metadata, batch_dim, max_batch_size)
            updated_config_dict = self._update_custom_configs(config_dict, dynamic_axes, trt_profile)

            optimize(model=self._module, dataloader=TorchDataloader(samples), **updated_config_dict)

        self._optimized = True

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

    def _update_custom_configs(self, config_dict, dynamic_axes, trt_profile):
        config_dict = copy.deepcopy(config_dict)

        if config_dict["custom_configs"] is None:
            config_dict["custom_configs"] = []

        onnx_config, trt_config, torch_trt_config = None, None, None
        for custom_config in config_dict["custom_configs"]:
            if isinstance(custom_config, OnnxConfig):
                onnx_config = custom_config
            elif isinstance(custom_config, TensorRTConfig):
                trt_config = custom_config
            elif isinstance(custom_config, TorchTensorRTConfig):
                torch_trt_config = custom_config

        if onnx_config is not None and onnx_config.dynamic_axes is None:
            onnx_config.dynamic_axes = dynamic_axes
        else:
            config_dict["custom_configs"].append(OnnxConfig(dynamic_axes=dynamic_axes))

        if trt_config is not None and trt_config.trt_profiles is None:
            trt_config.trt_profiles = [trt_profile]
            if trt_config.run_max_batch_size_search is None:
                trt_config.run_max_batch_size_search = True
        else:
            config_dict["custom_configs"].append(
                TensorRTConfig(trt_profiles=[trt_profile], run_max_batch_size_search=True)
            )

        if torch_trt_config is not None and torch_trt_config.trt_profiles is None:
            torch_trt_config.trt_profiles = [trt_profile]
            if torch_trt_config.run_max_batch_size_search is None:
                torch_trt_config.run_max_batch_size_search = True
        else:
            config_dict["custom_configs"].append(
                TorchTensorRTConfig(trt_profiles=[trt_profile], run_max_batch_size_search=True)
            )

        return config_dict


class RecordAndOptimizeModule(RecordModule):
    """Module that records samples for optimization and runs optimization."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module.

        If enough samples are collected for all registered modules, optimize them.
        """
        output = super().__call__(*args, **kwargs)
        if not self._optimized and module_registry.check_all_ready():
            module_registry.optimize()
        return output


class OptimizedModule(BaseModule):
    """Module that runs the optimized module."""

    def __init__(
        self, *args, strategy: Optional[RuntimeSearchStrategy] = None, activate_runners: bool = True, **kwargs
    ) -> None:
        """Initialize OptimizedModule.

        Load the optimized module from the Model Navigator workspace
        and get the runner according to the configured strategy.
        """
        super().__init__(*args, **kwargs)

        self._module.to("cpu")
        if torch.cuda.is_available():
            # empty torch cuda buffers after moving module to cpu i.e. more free vram
            torch.cuda.empty_cache()

        self._packages = []
        self._runners = {}

        strategy = strategy or inplace_config.strategy

        for package_workspace in self._workspace.iterdir():
            package = load_from_workspace(package_workspace)
            package.load_source_model(self._module)

            runner = package.get_runner(return_type=TensorType.TORCH, strategy=strategy)
            pytree_metadata = package.status.input_metadata.pytree_metadata

            self._packages.append(package)
            self._runners[pytree_metadata] = runner

        if activate_runners:
            self.activate_runners()
            self.runner_active = True
        else:
            self.runner_active = False

    def __del__(self):
        """Deactivate runners and delete packages."""
        if hasattr(self, "runner_active") and self.runner_active:
            self.deactivate_runners()

    def activate_runners(self):
        """Activate all runners."""
        for runner in self._runners.values():
            runner.activate()

    def deactivate_runners(self):
        """Deactivate all runners."""
        for runner in self._runners.values():
            runner.deactivate()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the call through the optimized module."""
        sample = (*args, kwargs)
        sample = self._input_mapping(sample)
        # TODO when only one runner is present, we can avoid the PyTreeMetadata
        pytree_metadata = PyTreeMetadata.from_sample(sample, TensorType.TORCH, prefix=PYTREE_METADATA_PREFIX)
        if pytree_metadata not in self._runners:
            raise ValueError(f"No runner found for {pytree_metadata}")
        runner = self._runners[pytree_metadata]
        if hasattr(
            runner, "inplace_infer"
        ):  # TODO this is a hack to avoid redundant flatten/unflatten for Torch runner
            output = runner.inplace_infer(*args, **kwargs)
        else:
            input_dict = runner.input_metadata.flatten_sample(sample)
            runner_output = runner.infer(input_dict)
            output = runner.output_metadata.unflatten_sample(runner_output)
        output = self._output_mapping(output)  # TODO better solution?
        return output

    @property
    def is_optimized(self) -> bool:
        """Check if the module is optimized."""
        return True


class PassthroughModule(BaseModule):
    """Module that passes through the original module."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass through the original module."""
        return self._module(*args, **kwargs)
