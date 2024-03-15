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
# noqa: D104
"""Public utilities for the Model Navigator API."""

import json
import logging
import pathlib
import tempfile
from typing import Any, Callable, Dict, Optional

from model_navigator.api.config import Framework, OptimizationProfile, SizedDataLoader
from model_navigator.commands.find_max_batch_size.find_max_batch_size import MaxBatchSizeFinder
from model_navigator.core.dataloader import to_numpy
from model_navigator.core.tensor import FRAMEWORK_TO_TENSOR_TYPE, PyTreeMetadata, TensorMetadata
from model_navigator.exceptions import ModelNavigatorError, ModelNavigatorProfilingError
from model_navigator.runners.registry import get_runner

logger_name = "model_navigator.api.utilities"
LOGGER = logging.getLogger(logger_name)
logging.basicConfig(level=logging.INFO)


class UnpackedDataloader:
    """A wrapper around a SizedDataLoader that applies a function to each sample.

    Args:
        dataloader: A SizedDataLoader.
        unpack_fn: A function that takes a sample and returns a new sample.

    Returns:
        An iterator over the samples in the dataloader with the unpack_fn applied.

    Example:
        >>> dataloader = [1, 2, 3]
        >>> unpacked_dataloader = UnpackedDataloader(dataloader, lambda x: x + 1)
        >>> # unpacked_dataloader is now [2, 3, 4]
    """

    def __init__(self, dataloader: SizedDataLoader, unpack_fn: Callable):
        """Initialize the UnpackedDataloader."""
        self._dataloader = dataloader
        self._unpack_fn = unpack_fn

    def __len__(self):
        """Return the number of samples in the dataloader."""
        return len(self._dataloader)

    def __iter__(self):
        """Return an iterator over the samples in the dataloader with the unpack_fn applied."""
        for sample in self._dataloader:
            yield self._unpack_fn(sample)


def find_max_batch_size_till_oom(
    framework: Framework,
    model: Any,
    dataloader: SizedDataLoader,
    batch_dim: int = 0,
    max_batch_size_search_limit: Optional[int] = None,
    runner_config: Optional[Dict] = None,
):
    """Find the maximum batch size for a model.

    Search is performed by running the model on the dataloader until an OOM error is encountered.

    Args:
        framework: The framework of the model.
        model: The model.
        dataloader: A SizedDataLoader.
        batch_dim: The batch dimension of the model.
        max_batch_size_search_limit: Limit the search for the maximum batch size to this value.
        runner_config: Additional runner configuration.
    """
    if framework == Framework.TORCH:
        runner_name = "TorchCUDA"
    elif framework == Framework.TENSORFLOW:
        runner_name = "TensorFlowCUDA"
    elif framework == Framework.ONNX:
        runner_name = "OnnxCUDA"
    elif framework == Framework.JAX:
        runner_name = "Jax"
    else:
        raise ModelNavigatorError(f"Unsupported {framework} for operation.")

    sample = next(iter(dataloader))

    pytree_metadata = PyTreeMetadata.from_sample(
        sample=sample, tensor_type=FRAMEWORK_TO_TENSOR_TYPE[framework], prefix="input"
    )
    input_metadata = TensorMetadata(pytree_metadata=pytree_metadata)

    profiling_sample = input_metadata.flatten_sample(sample=sample)
    profiling_sample = {name: to_numpy(tensor, from_framework=framework) for name, tensor in profiling_sample.items()}

    for name, tensor in profiling_sample.items():
        shape = list(tensor.shape)
        shape[batch_dim] = -1
        input_metadata.add(name=name, shape=shape, dtype=tensor.dtype)

    if runner_config is None:
        runner_config = {}

    optimization_profile = OptimizationProfile(
        max_batch_size=max_batch_size_search_limit,
        window_size=1,
        stabilization_windows=1,
        min_trials=1,
        max_trials=1,
        throughput_cutoff_threshold=-2,
    )

    with tempfile.NamedTemporaryFile() as temp_file:
        results_path = pathlib.Path(temp_file.name)

        runner = get_runner(runner_name)(
            model=model,
            input_metadata=input_metadata,
            output_metadata=None,
            enable_timer=True,
            **runner_config,
        )  # pytype: disable=not-instantiable
        try:
            LOGGER.info("Starting max batch size search.")
            MaxBatchSizeFinder(
                profile=optimization_profile,
                batch_dim=batch_dim,
                results_path=results_path,
            ).run(
                runner=runner,
                profiling_sample=profiling_sample,
                sample_id=0,
            )
        finally:
            if results_path.is_file():
                with open(results_path) as file:
                    max_bs_line = file.readlines()[-1]
                    results_dict = json.loads(max_bs_line)
                    if "batch_size" in results_dict:
                        LOGGER.info(f"Max batch size: {results_dict['batch_size']}")
                    else:
                        raise ModelNavigatorProfilingError("Max batch size not found.")
            else:
                raise ModelNavigatorProfilingError("Max batch size not found.")
        return results_dict["batch_size"]
