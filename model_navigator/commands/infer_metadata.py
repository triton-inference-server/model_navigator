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
"""Inputs and outputs metadata commands."""

import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from model_navigator.api.config import OptimizationProfile, SizedDataLoader, SizedIterable, TensorRTProfile
from model_navigator.commands.base import Command, CommandOutput, CommandStatus
from model_navigator.commands.execution_context import ExecutionContext
from model_navigator.core.dataloader import extract_sample, load_samples, to_numpy, validate_sample_input
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import FRAMEWORK_TO_TENSOR_TYPE, PyTreeMetadata, TensorMetadata, TensorSpec
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework, is_torch_available
from model_navigator.frameworks.onnx.utils import get_onnx_io_names
from model_navigator.frameworks.tensorrt.utils import get_tensorrt_io_names
from model_navigator.runners.utils import get_format_default_runners
from model_navigator.utils import module
from model_navigator.utils.format_helpers import FRAMEWORK2BASE_FORMAT

torch = module.lazy_import("torch")


def _extract_axes_shapes(
    dataloader: Union[SizedDataLoader, Iterator],
    pytree_metadata: PyTreeMetadata,
    input_names: Sequence[str],
    input_ndims: Sequence[int],
    num_samples: int,
    framework: Framework,
    check_len: bool = True,
) -> Dict[str, Dict[int, List[int]]]:
    assert not (check_len) or isinstance(
        dataloader, (SizedIterable, Sequence)
    ), "dataloader is not an instance of SizedDataLoader, unable to check length."

    axes_shapes = {name: {ax: [] for ax in range(ndim)} for name, ndim in zip(input_names, input_ndims)}
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            LOGGER.warning(f"{len(dataloader)=}, but more samples found.")
            break
        validate_sample_input(sample, FRAMEWORK_TO_TENSOR_TYPE[framework])
        sample = {n: to_numpy(t, framework) for n, t in pytree_metadata.flatten_sample(sample).items()}
        for name, tensor in sample.items():
            for k, dim in enumerate(tensor.shape):
                axes_shapes[name][k].append(dim)

    if check_len:
        assert i + 1 >= len(dataloader), f"{len(dataloader)=}, but only {i + 1} samples found."

    return axes_shapes


def _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, dtypes):
    metadata = TensorMetadata(pytree_metadata=pytree_metadata)
    for name, axes in axes_shapes.items():
        tensor_shape = []
        for ax, shapes in axes.items():
            if ax == batch_dim or min(shapes) != max(shapes):
                tensor_shape.append(-1)
            else:
                tensor_shape.append(shapes[0])
        metadata.add(name, tuple(tensor_shape), dtypes[name])
    return metadata


def _extract_max_batch_size(axes_shapes: Dict[str, Dict[int, List[int]]], batch_dim: Optional[int]) -> int:
    if batch_dim is not None:
        return max(list(axes_shapes.values())[0][batch_dim])
    return 0


def _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim, config_max_batch_size=None):
    trt_profile = TensorRTProfile()
    for name, axes in axes_shapes.items():
        min_opt_max = []
        for ax, shapes in axes.items():
            if ax == batch_dim:  # min bs = 1
                if config_max_batch_size and (config_max_batch_size < max(shapes)):
                    raise ModelNavigatorUserInputError(
                        f"Given configuration maximum batch size ({config_max_batch_size}) "
                        f"is smaller than the encountered batch size ({max(shapes)})."
                    )
                max_batch_size = config_max_batch_size or max(shapes)
                min_opt_max.append((1, int(np.median(shapes)), max_batch_size))
            else:
                min_opt_max.append((min(shapes), int(np.median(shapes)), max(shapes)))
        if min_opt_max:
            trt_profile.add(name, *list(zip(*min_opt_max)))
    return trt_profile


def _assert_all_inputs_have_same_pytree_metadata(
    dataloader: Union[SizedDataLoader, Iterator],
    pytree_metadata: PyTreeMetadata,
) -> bool:
    for sample in dataloader:
        if not pytree_metadata.is_compatible_with(sample):
            raise ModelNavigatorUserInputError(
                f"All inputs must have the same structure.\n"
                f"Input structure: {pytree_metadata}\n"
                f"Sample: {sample}."
            )


class InferInputMetadata(Command, is_required=True):
    """Command to collect model inputs metadata."""

    def _run(
        self,
        model: Union[object, pathlib.Path],
        framework: Framework,
        dataloader: SizedDataLoader,
        optimization_profile: OptimizationProfile,
        _input_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute the InferInputMetadata command.

        Args:
            framework: Framework of model to run inference
            model: A model object or path to file
            dataloader: Dataloader for providing samples
            optimization_profile: Optimization profile
            _input_names: Name of model inputs
            batch_dim: Location of batch dimension in data samples

        Returns:
            CommandOutput object
        """
        sample = next(iter(dataloader))
        validate_sample_input(sample, FRAMEWORK_TO_TENSOR_TYPE[framework])
        if framework == Framework.ONNX:
            if _input_names is not None:
                LOGGER.warning("ONNX input names are not supported yet. `input_names` will be ignored.")
            _input_names, _ = get_onnx_io_names(model)
        elif framework == Framework.TENSORRT:
            _input_names, _ = get_tensorrt_io_names(model)

        pytree_metadata = PyTreeMetadata.from_sample(
            sample, tensor_type=FRAMEWORK_TO_TENSOR_TYPE[framework], names=_input_names, prefix="input"
        )
        _assert_all_inputs_have_same_pytree_metadata(dataloader, pytree_metadata)
        input_sample = {}
        input_dtypes = {}
        for n, t in pytree_metadata.flatten_sample(sample).items():
            input_sample[n] = to_numpy(t, framework)

            # TODO: Remove this check once torch.bfloat16 is supported
            if not is_torch_available() or t.dtype != torch.bfloat16:
                input_dtypes[n] = input_sample[n].dtype
            else:
                input_dtypes[n] = torch.bfloat16
        input_names = list(input_sample.keys())

        input_ndims = [t.ndim for t in input_sample.values()]
        num_samples = len(dataloader)
        axes_shapes = _extract_axes_shapes(
            dataloader, pytree_metadata, input_names, input_ndims, num_samples, framework
        )
        dataloader_max_batch_size = _extract_max_batch_size(axes_shapes, batch_dim)
        dataloader_trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim)
        input_metadata = _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, input_dtypes)

        if optimization_profile.dataloader:
            pd_sample = next(iter(optimization_profile.dataloader))
            pd_input_sample = extract_sample(pd_sample, input_metadata, framework)
            pd_input_ndims = [t.ndim for t in pd_input_sample.values()]
            pd_input_dtypes = {n: t.dtype for n, t in pd_input_sample.items()}
            if pd_input_ndims != input_ndims:
                raise ModelNavigatorUserInputError(
                    "Provided performance dataloader does not match dataset dataloader size."
                )

            if pd_input_dtypes != input_dtypes:
                raise ModelNavigatorUserInputError(
                    "Provided performance dataloader does not match dataset dataloader data types."
                )

            self._validate_performance_dataloader_trt_profiles(
                optimization_profile=optimization_profile,
                pytree_metadata=input_metadata.pytree_metadata,
                input_names=input_names,
                input_ndims=pd_input_ndims,
                framework=framework,
                batch_dim=batch_dim,
                dataloader_trt_profile=dataloader_trt_profile,
            )

        return CommandOutput(
            status=CommandStatus.OK,
            output={
                "input_metadata": input_metadata,
                "dataloader_trt_profile": dataloader_trt_profile,
                "dataloader_max_batch_size": dataloader_max_batch_size,
            },
        )

    def _validate_performance_dataloader_trt_profiles(
        self,
        optimization_profile: OptimizationProfile,
        pytree_metadata,
        input_names,
        input_ndims,
        framework,
        batch_dim,
        dataloader_trt_profile,
    ):
        axes_shapes = _extract_axes_shapes(
            dataloader=optimization_profile.dataloader,
            pytree_metadata=pytree_metadata,
            input_names=input_names,
            input_ndims=input_ndims,
            num_samples=1,
            framework=framework,
            check_len=False,
        )
        performance_dataloader_trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim)
        for name, shape in performance_dataloader_trt_profile.items():
            dshape = dataloader_trt_profile[name]
            if shape.min < dshape.min or shape.max > dshape.max:
                raise ModelNavigatorUserInputError(
                    """Provided performance dataloader has invalid shape against the dataset dataloader."""
                    f""" Performance dataloader shape for input `{name}` is min: {shape.min}, max: {shape.max}."""
                    f""" Dataset dataloader shape for input `{name}` is min: {dshape.min}, max: {dshape.max}."""
                )


class InferOutputMetadata(Command, is_required=True):
    """Command to collect model outputs  metadata."""

    def _run(
        self,
        framework: Framework,
        model: Union[object, pathlib.Path],
        dataloader: SizedDataLoader,
        input_metadata: TensorMetadata,
        workspace: Workspace,
        verbose: bool,
        _output_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
    ) -> CommandOutput:
        """Execute the InferOutputMetadata command.

        Args:
            framework: Framework of model to run inference
            model: A model object or path to file
            dataloader: Dataloader for providing samples
            input_metadata: Model inputs metadata
            workspace: Working directory where command should be executed
            verbose: Enable verbose logging
            _output_names: Name of model outputs
            batch_dim: Location of batch dimension in data samples

        Returns:
            CommandOutput object
        """
        if framework == Framework.ONNX:
            if _output_names is not None:
                LOGGER.warning("ONNX output names are not supported yet. `output_names` will be ignored.")
            _, _output_names = get_onnx_io_names(model)
            temp_output_metadata = TensorMetadata({out_name: TensorSpec(out_name, ()) for out_name in _output_names})
        elif framework == Framework.TENSORRT:
            _, _output_names = get_tensorrt_io_names(model)
            temp_output_metadata = TensorMetadata({out_name: TensorSpec(out_name, ()) for out_name in _output_names})
        else:
            temp_output_metadata = None
        runner = get_format_default_runners(FRAMEWORK2BASE_FORMAT[framework])[0](
            model=model,
            input_metadata=input_metadata,
            output_metadata=temp_output_metadata,
            disable_fallback=False,
        )  # pytype: disable=not-instantiable

        profiling_sample = load_samples("profiling_sample", workspace.path, batch_dim)[0]
        conversion_samples = load_samples("conversion_sample", workspace.path, batch_dim)

        with runner, ExecutionContext(workspace=workspace, verbose=verbose):
            outputs = runner.infer(profiling_sample)
            pytree_metadata = PyTreeMetadata.from_sample(
                outputs, tensor_type=FRAMEWORK_TO_TENSOR_TYPE[framework], names=_output_names, prefix="output"
            )
            output_sample = {n: to_numpy(t, framework) for n, t in pytree_metadata.flatten_sample(outputs).items()}
            output_names = list(output_sample.keys())
            output_generator = (runner.infer(sample) for sample in conversion_samples)

            output_ndims = [t.ndim for t in output_sample.values()]
            output_dtypes = {n: t.dtype for n, t in output_sample.items()}
            num_samples = len(dataloader)
            axes_shapes = _extract_axes_shapes(
                output_generator, pytree_metadata, output_names, output_ndims, num_samples, framework, check_len=False
            )

        output_metadata = _get_metadata_from_axes_shapes(pytree_metadata, axes_shapes, batch_dim, output_dtypes)

        return CommandOutput(
            status=CommandStatus.OK,
            output={"output_metadata": output_metadata},
        )
