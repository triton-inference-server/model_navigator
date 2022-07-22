# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from polygraphy.backend.onnxrt import SessionFromOnnx
from polygraphy.backend.trt import Profile

from model_navigator.framework_api.commands.core import Command, CommandType
from model_navigator.framework_api.common import Sample, SizedDataLoader, TensorMetadata
from model_navigator.framework_api.exceptions import ExecutionContext, UserError
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.onnx import OnnxrtRunner
from model_navigator.framework_api.utils import (
    Framework,
    Status,
    extract_sample,
    get_available_onnx_providers,
    get_trt_profile_from_trt_dynamic_axes,
    sample_to_tuple,
    validate_sample_input,
)
from model_navigator.tensor import TensorSpec

if TYPE_CHECKING:
    from model_navigator.framework_api.package_descriptor import PackageDescriptor


def _extract_axes_shapes(
    dataloader: SizedDataLoader,
    input_names: Sequence[str],
    input_ndims: Sequence[int],
    num_samples: int,
    framework: Framework,
    check_len: bool = True,
) -> Dict[str, Dict[int, List[int]]]:
    axes_shapes = {name: {ax: [] for ax in range(ndim)} for name, ndim in zip(input_names, input_ndims)}
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            LOGGER.warning(f"{len(dataloader)=}, but more samples found.")
            break
        validate_sample_input(sample, framework)
        sample = extract_sample(sample, input_names, framework)
        for name, tensor in sample.items():
            for k, dim in enumerate(tensor.shape):
                axes_shapes[name][k].append(dim)

    if check_len:
        assert i + 1 >= len(dataloader), f"{len(dataloader)=}, but only {i + 1} samples found."

    return axes_shapes


def _extract_max_batch_size(axes_shapes: Dict[str, Dict[int, List[int]]], batch_dim: Optional[int]) -> int:
    if batch_dim is not None:
        return max(list(axes_shapes.values())[0][batch_dim])
    return 0


def _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim):
    trt_profile = Profile()
    for name, axes in axes_shapes.items():
        min_max_opt = []
        for ax, shapes in axes.items():
            if ax == batch_dim:  # min bs = 1
                min_max_opt.append((1, int(np.median(shapes)), max(shapes)))
            else:
                min_max_opt.append((min(shapes), int(np.median(shapes)), max(shapes)))
        trt_profile.add(name, *list(zip(*min_max_opt)))
    return trt_profile


def _verify_user_trt_profile(user_profile: Profile, dl_profile: Profile):
    for name, user_shape_tuple in user_profile.items():
        dl_shape_tuple = dl_profile.get(name, None)
        if dl_shape_tuple is None:
            raise ValueError(f"TRT dynamic axes not specified for axis `{name}`.")
        for attr in ("min", "opt", "max"):
            user_shapes, dl_shapes = getattr(user_shape_tuple, attr), getattr(dl_shape_tuple, attr)
            if len(user_shapes) != len(dl_shapes):
                raise ValueError(f"Incorrect number of dimensions in TRT dynamic axes for `{name}`.")
            for ax, (user_shape, dl_shape) in enumerate(zip(user_shapes, dl_shapes)):
                if attr == "min":
                    if user_shape > dl_shape:
                        raise ValueError(
                            f"In TRT dynamic axes min shape for `{name}` on axis `{ax}` is {user_shape} but found {dl_shape} in the dataloader."
                        )
                if attr == "max":
                    if user_shape < dl_shape:
                        raise ValueError(
                            f"In TRT dynamic axes max shape for `{name}` on axis `{ax}` is {user_shape} but found {dl_shape} in the dataloader."
                        )


def _get_metadata_from_axes_shapes(axes_shapes, batch_dim, dtypes):
    metadata = TensorMetadata()
    for name, axes in axes_shapes.items():
        tensor_shape = []
        for ax, shapes in axes.items():
            if ax == batch_dim or min(shapes) != max(shapes):
                tensor_shape.append(-1)
            else:
                tensor_shape.append(shapes[0])
        metadata.add(name, tuple(tensor_shape), dtypes[name])
    return metadata


def _verify_and_update_user_dynamic_axes_(dynamic_axes, input_metadata):
    for name, axes in dynamic_axes.items():
        axes = list(axes)
        tensor_spec = input_metadata.get(name, None)
        if tensor_spec is None:
            return
        shape = list(tensor_spec.shape)
        for ax in axes:
            shape[ax] = -1  # update
        input_metadata[name] = TensorSpec(name, tuple(shape), tensor_spec.dtype)
        for ax, d in enumerate(tensor_spec.shape):
            if d == -1 and ax not in axes:  # verify
                raise ValueError(
                    f"In tensor `{name}` axis `{ax}` is not set as dynamic axes but is dynamic in the dataloader."
                )


class InferInputMetadata(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(name="Infer input metadata.", command_type=CommandType.INFER_MODEL_INPUT, requires=requires)

    @staticmethod
    def get_output_name():
        return (
            "input_metadata",
            "trt_profile",
            "max_batch_size",
        )

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        if self.status == Status.OK:
            package_descriptor.navigator_status.input_metadata = self.output[0]
            package_descriptor.navigator_status.trt_profile = self.output[1]

    def __call__(
        self,
        model: Union[object, Path],
        framework: Framework,
        dataloader: SizedDataLoader,
        _input_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        trt_dynamic_axes: Optional[Dict[str, Dict[int, Tuple[int, int, int]]]] = None,
        **kwargs,
    ) -> Tuple[TensorMetadata, Profile, int]:

        sample = next(iter(dataloader))
        validate_sample_input(sample, framework)
        input_tuple = sample_to_tuple(sample)
        input_names = _input_names
        input_metadata = None
        if input_names is None:
            if framework == Framework.ONNX:
                # pytype: disable=attribute-error
                onnx_runner = OnnxrtRunner(
                    SessionFromOnnx(model.as_posix(), providers=get_available_onnx_providers(exclude_trt=True))
                )
                # pytype: enable=attribute-error
                with onnx_runner:
                    input_metadata = TensorMetadata.from_polygraphy_tensor_metadata(onnx_runner.get_input_metadata())
                    input_names = tuple(input_metadata.keys())
            elif isinstance(sample, Mapping):
                input_names = tuple(sample.keys())
            else:
                input_names = tuple(f"input__{i}" for i in range(len(input_tuple)))

        input_sample = extract_sample(sample, input_names, framework)
        input_ndims = [t.ndim for t in input_sample.values()]
        input_dtypes = {n: t.dtype for n, t in input_sample.items()}
        num_samples = len(dataloader)
        axes_shapes = _extract_axes_shapes(dataloader, input_names, input_ndims, num_samples, framework)
        max_batch_size = _extract_max_batch_size(axes_shapes, batch_dim)

        dataloader_trt_profile = _get_trt_profile_from_axes_shapes(axes_shapes, batch_dim)
        if trt_dynamic_axes is None:
            trt_profile = dataloader_trt_profile

            LOGGER.warning(
                f"No TRT (min, opt, max) values for axes provided. Using values derived from the dataloader: {trt_profile}."
            )
        else:
            trt_profile = get_trt_profile_from_trt_dynamic_axes(trt_dynamic_axes)
            _verify_user_trt_profile(trt_profile, dataloader_trt_profile)

        input_metadata = _get_metadata_from_axes_shapes(axes_shapes, batch_dim, input_dtypes)
        if dynamic_axes is None:
            LOGGER.warning(f"No dynamic axes provided. Using values derived from the dataloader: {input_metadata}")
        else:
            _verify_and_update_user_dynamic_axes_(dynamic_axes, input_metadata)

        return input_metadata, trt_profile, max_batch_size


class InferOutputMetadata(Command):
    def __init__(self, requires: Tuple[Command, ...] = ()):
        super().__init__(
            name="Infer output metadata.",
            command_type=CommandType.INFER_MODEL_OUTPUT,
            requires=requires,
        )

    @staticmethod
    def get_output_name():
        return "output_metadata"

    def _update_package_descriptor(self, package_descriptor: "PackageDescriptor", **kwargs) -> None:
        if self.status == Status.OK:
            package_descriptor.navigator_status.output_metadata = self.output

    def __call__(
        self,
        framework: Framework,
        model: Union[object, Path],
        dataloader: SizedDataLoader,
        profiling_sample: Sample,
        conversion_samples: List[Sample],
        input_metadata: TensorMetadata,
        target_device: Optional[str] = None,
        _output_names: Optional[Tuple[str, ...]] = None,
        dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]] = None,
        forward_kw_names: Optional[Tuple[str, ...]] = None,
        batch_dim: Optional[int] = None,
        **kwargs,
    ) -> TensorMetadata:

        if framework == Framework.PYT:
            from model_navigator.framework_api.runners.pyt import PytRunner

            runner = PytRunner(
                model, input_metadata, _output_names, target_device=target_device, forward_kw_names=forward_kw_names
            )
        elif framework == Framework.TF2:
            from model_navigator.framework_api.runners.tf import TFRunner

            runner = TFRunner(model, input_metadata, _output_names)
        elif framework == Framework.ONNX:
            # pytype: disable=attribute-error
            runner = OnnxrtRunner(
                SessionFromOnnx(model.as_posix(), providers=get_available_onnx_providers(exclude_trt=True))
            )
            # pytype: enable=attribute-error
        else:
            raise UserError(f"Unknown framework: {framework.value}")

        with runner, ExecutionContext():
            output_sample = runner.infer(profiling_sample)
            output_names = list(output_sample.keys())
            output_generator = (runner.infer(sample) for sample in conversion_samples)

            output_ndims = [t.ndim for t in output_sample.values()]
            output_dtypes = {n: t.dtype for n, t in output_sample.items()}
            num_samples = len(dataloader)
            axes_shapes = _extract_axes_shapes(
                output_generator, output_names, output_ndims, num_samples, framework, check_len=False
            )

        output_metadata = _get_metadata_from_axes_shapes(axes_shapes, batch_dim, output_dtypes)
        if dynamic_axes is None:
            LOGGER.warning(f"No dynamic axes provided. Using values derived from the dataloader: {output_metadata}")
        else:
            _verify_and_update_user_dynamic_axes_(dynamic_axes, output_metadata)

        return output_metadata
