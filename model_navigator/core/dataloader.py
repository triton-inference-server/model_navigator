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
"""Dataloader and samples core functionality."""

import math
import pathlib
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from model_navigator.api.config import Sample, TensorType
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata, is_tensor
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework
from model_navigator.utils import module
from model_navigator.utils.common import PYTHON_PRIMITIVE_TYPES

torch = module.lazy_import("torch")

SAMPLE_FILE_SUFFIX = ".npz"


class IndiciesFilteredDataloader:
    """Dataloader that filters indices."""

    def __init__(self, dataloader: Any, indicies: List[int]):
        """Initialize IndiciesFilteredDataloader.

        Args:
            dataloader: A dataloader to filter
            indicies: A list of indices to filter
        """
        self._dataloader = dataloader
        self._indicies = indicies

    def __iter__(self):
        """Iterate over samples."""
        for idx, sample in enumerate(self._dataloader):
            if idx in self._indicies:
                yield sample

    def __len__(self):
        """Get number of samples."""
        return len(self._indicies)


class SortedSamplesLoader:
    """Dataloader that loads samples from directory."""

    def __init__(self, samples_dirpath: pathlib.Path, batch_dim: Optional[int] = None):
        """Initialize SamplesLoader.

        Args:
            samples_dirpath: Path to samples directory
            batch_dim: Batch dimension
        """
        self._samples_paths = self._samples_files(samples_dirpath)
        self._batch_dim = batch_dim

    def __getitem__(self, idx: int) -> Sample:
        """Get sample for given index.

        Args:
            idx: Index of sample to get

        Returns:
            Sample data
        """
        sample_filepath = self._samples_paths[idx]
        sample = {}
        with np.load(sample_filepath.as_posix()) as data:
            for k, v in data.items():
                if self._batch_dim is not None:
                    v = np.expand_dims(v, self._batch_dim)
                sample[k] = v
        return sample

    def __len__(self) -> int:
        """Get number of samples.

        Returns:
            Number of samples
        """
        return len(self._samples_paths)

    def _samples_files(self, samples_dirpath: pathlib.Path):
        """Collect sample files from directory in sorted order.

        Args:
            samples_dirpath: Path to samples directory
        """
        files = samples_dirpath.iterdir()
        files = [f for f in files if f.suffix == SAMPLE_FILE_SUFFIX]
        files = sorted(files, key=lambda f: int(f.name.split(".")[0]))

        return files


def load_samples(
    samples_name: str, workspace: Union[pathlib.Path, str], batch_dim: Optional[int]
) -> SortedSamplesLoader:
    """Load samples for provided name.

    Args:
        samples_name: Name of samples to load
        workspace: Working directory
        batch_dim: Position of batch dimension

    Returns:
        List of data samples
    """
    if isinstance(workspace, str):
        workspace = pathlib.Path(workspace)
    samples_type = samples_name.split("_")[0]
    samples_dirname = "model_output" if samples_name.split("_")[-1] == "output" else "model_input"
    samples_dirpath = workspace / samples_dirname / samples_type

    return SortedSamplesLoader(samples_dirpath, batch_dim)


def samples_to_npz(
    samples: Iterable[Sample],
    path: pathlib.Path,
    batch_dim: Optional[int],
    *,
    num_samples: Optional[int] = None,
    metadata: Optional[TensorMetadata] = None,
    framework: Optional[Framework] = None,
    raise_on_error: bool = True,
) -> None:
    """Save samples to .npz files. Each sample is saved to `path/{sample index}.npz` file.

    Args:
        samples (List[Sample]): Samples to save.
        path (Path): Output directory.
        batch_dim (Optional[int]): Batch dimension
        num_samples (Optional[int], optional): Number of samples to save. Defaults to None.
        metadata (Optional[TensorMetadata], optional): Metadata of the samples. Defaults to None.
        framework (Framework): Model framework. Defaults to None.
        raise_on_error (bool, optional): If True raise an error when sample is invalid. Defaults to True.
    """
    path.mkdir(parents=True, exist_ok=True)
    if num_samples is None:
        assert hasattr(samples, "__len__")
        num_samples = len(samples)

    for i, sample in enumerate(samples):
        if metadata is not None:
            assert framework is not None
            sample = extract_sample(sample, metadata, framework)
        sample = extract_bs1(sample, batch_dim)
        squeezed_sample = {}
        for name, tensor in sample.items():
            if batch_dim is not None:
                tensor = tensor.squeeze(batch_dim)

            _validate_tensor(tensor, raise_on_error=raise_on_error)

            squeezed_sample[name] = tensor

        filename = _sample_filename(idx=i, num_samples=num_samples)
        file_path = path / filename
        np.savez(file_path.as_posix(), **squeezed_sample)


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
        framework: A framework for which extraction is performed

    Returns:
        A formatted sample data
    """
    sample = input_metadata.flatten_sample(sample)
    sample = {n: to_numpy(t, framework) for n, t in sample.items()}
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


def get_tensor_type_name(tensor_type: TensorType) -> str:
    """Obtain name of tensor type for given framework.

    Args:
        tensor_type: A framework for which tensor type name has to be obtained

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
        # If torch.bfloat16 is used in dataloader perform lossless upcast to torch.float32 before conversion to numpy
        # TODO: remove to(torch.float32) once torch.bfloat16 is supported
        return (
            tensor.detach().cpu().numpy()
            if tensor.dtype != torch.bfloat16
            else tensor.to(torch.float32).detach().cpu().numpy()
        )
    if from_framework == Framework.TENSORFLOW:
        return tensor.numpy()
    return np.asarray(tensor)


def validate_sample_input(sample: Any, tensor_type: TensorType = TensorType.NUMPY) -> None:
    """Validate if sample input is correct input object.

    Args:
        sample: A sample to validate
        tensor_type: A framework for which the validation is performed

    Raises:
        ModelNavigatorUserInputError when provided sample if not a valid input object

    """
    if not _is_valid_io(sample, tensor_type):
        tensor_type = get_tensor_type_name(tensor_type)
        raise ModelNavigatorUserInputError(
            f"Invalid sample type. Sample must be of type Union[{tensor_type}, "
            f"Iterable[{tensor_type}], Mapping[str, {tensor_type}]]. Dataloader returned {sample}."
        )  # TODO fix this message


def validate_sample_output(sample, tensor_type: TensorType = TensorType.NUMPY):
    """Validate if sample output is correct output object.

    Args:
        sample: A sample to validate
        tensor_type: A framework for which the validation is performed

    Raises:
        ModelNavigatorUserInputError when provided sample if not a valid output object

    """
    if not _is_valid_io(sample, tensor_type):
        tensor_type = get_tensor_type_name(tensor_type)
        raise ModelNavigatorUserInputError(
            f"Invalid model output type. Output must be of type Union[{tensor_type}, "
            f"Iterable[{tensor_type}]], Mapping[str, {tensor_type}]]. Model returned {sample}."
        )  # TODO fix this message


def get_default_output_names(num_output: int) -> List:
    """Generate list of default output names.

    Args:
        num_output (int): Number of outputs.

    Returns:
        List: Default names.
    """
    return [f"output__{i}" for i in range(num_output)]


def _sample_filename(idx: int, num_samples: int) -> str:
    """Create filename for data sample with given index.

    Args:
        idx: Index of sample to store
        num_samples: Number of samples that will be generated

    Returns:
        String with filename
    """
    if idx > num_samples:
        raise ValueError("Incorrect value. `num_samples` must be greater than `idx`.")

    num_digits = math.floor(math.log10(num_samples)) + 1
    sample_name = str(idx).zfill(num_digits)
    filename = f"{sample_name}{SAMPLE_FILE_SUFFIX}"

    return filename


def _validate_tensor(tensor: np.ndarray, *, raise_on_error: bool = True):
    if any(np.isnan(tensor.flatten())):
        message = "Tensor data contains `NaN` value. Please verify the dataloader and model."
        if raise_on_error:
            raise ModelNavigatorUserInputError(message)
        else:
            LOGGER.warning(message)

    if any(np.isinf(tensor.flatten())):
        message = "Tensor data contains `inf` value. Please verify the dataloader and model."
        if raise_on_error:
            raise ModelNavigatorUserInputError(message)
        else:
            LOGGER.warning(message)


def _is_valid_io(sample: Any, tensor_type: TensorType) -> bool:
    """Validate if provided sample is correct I/O object.

    Args:
        sample: A sample to validate
        tensor_type: A framework for which the validation is performed

    Returns:
        True if sample is valid I/O, False otherwise
    """
    if is_tensor(sample, tensor_type) or isinstance(sample, PYTHON_PRIMITIVE_TYPES):
        return True
    if isinstance(sample, Mapping):
        for name, tensor in sample.items():
            if not isinstance(name, str):
                return False
            if not _is_valid_io(tensor, tensor_type):
                return False
        return True
    elif isinstance(sample, Sequence):
        for tensor in sample:
            if not _is_valid_io(tensor, tensor_type):
                return False
        return True
    return False
