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
"""Dataloader definition and helpers module."""
import pathlib
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from model_navigator.api.config import Sample, TensorType
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework


def to_numpy(tensor: Any, from_framework: Framework) -> np.ndarray:
    """Convert tensor to numpy array.

    Args:
        tensor: A tensor to convert
        from_framework: Framework from which convert to numpy

    Returns:
        Data in form of np.ndarray
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if from_framework == Framework.TORCH:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.numpy()


def sample_to_tuple(input: Any) -> Tuple[Any, ...]:
    """Convert sample to tuple.

    Args:
        input: A sample to convert

    Returns:
        Sample in form of tuple
    """
    if isinstance(input, Sequence):
        return tuple(input)
    if isinstance(input, Mapping):
        return tuple(input.values())
    return (input,)


def extract_sample(sample, input_metadata, framework: Framework) -> Sample:
    """Extract samples for inputs.

    Args:
        sample: A dataloader sample to extract from
        input_metadata: An input metadata
        framework: A framework for which extraction is peformed

    Returns:
        A formatted sample data
    """
    sample = sample_to_tuple(sample)
    sample = {n: to_numpy(t, framework) for n, t in zip(input_metadata, sample)}
    return sample


def extract_bs1(sample: Sample, batch_dim: Optional[int]) -> Sample:
    """Extract sample with batch size 1.

    Args:
        sample: A sample to extract
        batch_dim: A place where batch is stored in sample

    Returns:
        A formatted sample data
    """
    if batch_dim is not None:
        return {name: tensor.take([0], batch_dim) for name, tensor in sample.items()}
    return sample


def is_tensor(tensor: Any, tensor_type: TensorType) -> bool:
    """Validate if provided object is a valid tensor.

    Args:
        tensor: An object to validate
        framework: A framework for which the object has to be tested

    Returns:
        True if object is a valid tensor, False otherwise
    """
    if tensor_type == TensorType.TORCH:
        import torch  # pytype: disable=import-error

        return torch.is_tensor(tensor) or isinstance(tensor, np.ndarray)
    elif tensor_type == TensorType.TENSORFLOW:
        import tensorflow  # pytype: disable=import-error

        return tensorflow.is_tensor(tensor) or isinstance(tensor, np.ndarray)
    else:
        return isinstance(tensor, np.ndarray)


def get_tensor_type_name(tensor_type: TensorType) -> str:
    """Obtain name of tensor type for given framework.

    Args:
        framework: A framework for which tensor type name has to be obtained

    Returns:
        Name of tensor type in form o string
    """
    if tensor_type == TensorType.TORCH:
        return "Union[torch.Tensor, numpy.ndarray]"
    elif tensor_type == TensorType.TENSORFLOW:
        return "Union[tensorflow.Tensor, numpy.ndarray]"
    elif tensor_type == TensorType.NUMPY:
        return "numpy.ndarray"
    else:
        raise ValueError(f"Unknown tensor type {tensor_type}")


def _is_valid_io(sample: Any, tensor_type: TensorType) -> bool:
    """Validate if provided sample is correct I/O object.

    Args:
        sample: A sample to validate
        framework: A framework for which the validation is performed

    Returns:
        True if sample is valid I/O, False otherwise
    """
    if is_tensor(sample, tensor_type):
        return True
    if isinstance(sample, Mapping):
        for tensor in sample.values():
            if not is_tensor(tensor, tensor_type):
                return False
        return True
    elif isinstance(sample, Iterable):
        for tensor in sample:
            if not is_tensor(tensor, tensor_type):
                return False
        return True
    return False


def validate_sample_input(sample: Any, tensor_type: TensorType = TensorType.NUMPY) -> None:
    """Validate if sample input is correct input object.

    Args:
        sample: A sample to validate
        framework: A framework for which the validation is performed

    Raises:
        ModelNavigatorUserInputError when provided sample if not a valid input object

    """
    if not _is_valid_io(sample, tensor_type):
        tensor_type = get_tensor_type_name(tensor_type)
        raise ModelNavigatorUserInputError(
            f"Invalid sample type. Sample must be of type Union[{tensor_type}, "
            f"Iterable[{tensor_type}], Mapping[str, {tensor_type}]]. Dataloader returned {sample}."
        )


def validate_sample_output(sample, tensor_type: TensorType = TensorType.NUMPY):
    """Validate if sample output is correct output object.

    Args:
        sample: A sample to validate
        framework: A framework for which the validation is performed

    Raises:
        ModelNavigatorUserInputError when provided sample if not a valid output object

    """
    if not _is_valid_io(sample, tensor_type):
        tensor_type = get_tensor_type_name(tensor_type)
        raise ModelNavigatorUserInputError(
            f"Invalid model output type. Output must be of type Union[{tensor_type}, "
            f"Iterable[{tensor_type}]], Mapping[str, {tensor_type}]]. Model returned {sample}."
        )


def load_samples(samples_name: str, workspace: Union[pathlib.Path, str], batch_dim: Optional[int]) -> List[Sample]:
    """Load samples for provided name.

    Args:
        samples_name: Name of samples to load
        workspace: Working directory
        batch_dim: Position of batch dimension

    Returns:
        List of data samples
    """
    if isinstance(workspace, str):
        workspace = Path(workspace)
    samples_type = samples_name.split("_")[0]
    samples_dirname = "model_output" if samples_name.split("_")[-1] == "output" else "model_input"
    samples_dirpath = workspace / samples_dirname / samples_type
    samples = []
    for sample_filepath in samples_dirpath.iterdir():
        sample = {}
        with np.load(sample_filepath.as_posix()) as data:
            for k, v in data.items():
                if batch_dim is not None:
                    v = np.expand_dims(v, batch_dim)
                    # v = numpy.repeat(v, max_batch_size, batch_dim)
                sample[k] = v
        samples.append(sample)

    return samples


def get_default_output_names(num_output: int) -> List:
    """Generate list of default output names.

    Args:
        num_output (int): Number of outputs.

    Returns:
        List: Default names.
    """
    return [f"output__{i}" for i in range(num_output)]


def expand_sample(sample: Sample, batch_dim: Optional[int], batch_size: Optional[int]) -> Sample:
    """Expand sample to a given batch size.

    Args:
        sample (Sample): Sample to be expanded.
        batch_dim (Optional[int]): Batch dimension.
        batch_size (int): Batch size.

    Returns:
        Sample: Expanded Sample.
    """
    if batch_dim is None:
        assert batch_size is None, f"Batching is disabled but batch size is not None: {batch_size}."
        return sample
    expanded_sample = {}
    for name, tensor in sample.items():
        expanded_sample[name] = tensor.repeat(batch_size, axis=batch_dim)
    return expanded_sample
