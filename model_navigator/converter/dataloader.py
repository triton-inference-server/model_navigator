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
import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from model_navigator.converter import DatasetProfileConfig
from model_navigator.converter.utils import navigator_subprocess
from model_navigator.exceptions import ModelNavigatorException
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.nav_package import NavPackage

LOGGER = logging.getLogger(__name__)


class Dataloader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        ...

    def __call__(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from self.__iter__()

    @property
    @abstractmethod
    def opt_shapes(self) -> Dict[str, List[int]]:
        ...

    @property
    @abstractmethod
    def min_shapes(self) -> Dict[str, List[int]]:
        ...

    @property
    @abstractmethod
    def max_shapes(self) -> Dict[str, List[int]]:
        ...


def _generate_random(dtype, shapes, value_range, rng) -> Iterable[Tuple]:
    if dtype.kind == "i":
        return rng.integers(value_range[0], value_range[1], size=shapes, dtype=dtype.type)
    elif dtype.kind == "f":
        return np.array(
            rng.standard_normal(size=shapes) * (value_range[1] - value_range[0]) + value_range[0],
            dtype=dtype.type,
        )
    else:
        raise ModelNavigatorException(f"Don't know how to generate random tensor for dtype={dtype}")


def _get_tf_signature(model_path):
    import tensorflow as tf  # pytype: disable=import-error

    def conv_shape(shape):
        xl = shape.as_list()
        for i, x in enumerate(xl):
            if not x:
                xl[i] = -1
        return tuple(xl)

    model = tf.saved_model.load(model_path.as_posix())

    try:
        concrete_func = model.signatures["serving_default"]

        # inspired by https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py#L205
        if concrete_func.structured_input_signature:
            input_args, input_kwargs = concrete_func.structured_input_signature
            input_names = list(input_kwargs)
            assert not input_args, f"Unsupported positional args in concrete function signature: args={input_args}"
            inputs = {
                name: TensorSpec(sig.name, conv_shape(sig.shape), np.dtype(sig.dtype.as_numpy_dtype))
                for name, sig in input_kwargs.items()
            }
        elif concrete_func._arg_keywords:  # pylint: disable=protected-access
            # For pure ConcreteFunctions we might have nothing better than _arg_keywords.
            assert concrete_func._num_positional_args in [0, 1]
            input_names = concrete_func._arg_keywords

            # TODO: Is this needed at all? Is this even correct?
            input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
            inputs = {
                name: TensorSpec(tensor.name, conv_shape(tensor.shape), np.dtype(tensor.dtype.as_numpy_dtype))
                for name, tensor in zip(input_names, input_tensors)
            }

        outputs = {
            tensor.name: TensorSpec(tensor.name, conv_shape(tensor.shape), np.dtype(tensor.dtype.as_numpy_dtype))
            for tensor in concrete_func.outputs
        }

        return ModelSignatureConfig(inputs=inputs, outputs=outputs)
    except Exception as e:
        LOGGER.error(e)
        raise


def extract_model_signature(model_path):
    # run in a separate process to avoid problems with global resource management
    with navigator_subprocess() as navigator:
        module = navigator.module("model_navigator.converter.dataloader")
        return module._get_tf_signature(model_path)


def _check_dynamic_dims(shape, inp):
    for x in shape:
        if x == -1:
            raise ModelNavigatorException(
                "Cannot construct default dataset profile: "
                f"too many dynamic axes in the model input {inp.name}: {list(inp.shape)}. "
                "Please provide a full dataset profile instead of max_batch_size."
            )


def _shapes_from_signature(model_signature, max_batch_size):
    min_shapes = {}
    for name, inp in model_signature.inputs.items():
        min_shape = list(inp.shape)
        min_shape[0] = 1
        _check_dynamic_dims(min_shape, inp)
        min_shapes[name] = min_shape

    max_shapes = {}
    for name, inp in model_signature.inputs.items():
        max_shape = list(inp.shape)
        max_shape[0] = max_batch_size
        _check_dynamic_dims(max_shape, inp)
        max_shapes[name] = max_shape
    return min_shapes, max_shapes


class RandomDataloader(Dataloader):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        model_signature_config: Optional[ModelSignatureConfig] = None,
        dataset_profile: Optional[DatasetProfileConfig] = None,
        max_batch_size: int = 1,
        random_seed: int = 0,
    ):
        self.dataset_profile = dataset_profile or DatasetProfileConfig()
        self.rng = np.random.default_rng(random_seed)
        if not self._dataset_profile_complete():
            LOGGER.debug("Dataset profile incomplete, reconstructing from model signature.")
            self._generate_default_profile(model_config, model_signature_config, max_batch_size)

    def _dataset_profile_complete(self):
        return None not in (
            self.dataset_profile.min_shapes,
            self.dataset_profile.max_shapes,
            self.dataset_profile.dtypes,
            self.dataset_profile.value_ranges,
        )

    def _generate_default_profile(self, model_config, model_signature, max_batch_size):
        if model_config is None and model_signature is None:
            raise ModelNavigatorException("Cannot reconstruct input signatures")

        if model_signature is None or model_signature.inputs is None or model_signature.outputs is None:
            model_signature = extract_model_signature(model_config.model_path)

        # dtypes are almost always missing, so do those first
        if self.dataset_profile.dtypes is None:
            dtypes = {}
            for name, inp in model_signature.inputs.items():
                dtypes[name] = inp.dtype
            self.dataset_profile.dtypes = dtypes
            LOGGER.info("Generated default dataset profile: dtypes=%s.", dtypes)

        if self.dataset_profile.value_ranges is None:

            def _get_default_value_range(spec: TensorSpec):
                return {"i": (0, 15), "f": (0.0, 1.0)}[spec.dtype.kind]

            value_ranges = {name: _get_default_value_range(spec) for name, spec in model_signature.inputs.items()}
            self.dataset_profile.value_ranges = value_ranges
            LOGGER.info(
                "Missing model input value ranges required during conversion. "
                "Use `value_ranges` config to define missing dataset profiles. "
                f"Used default values_ranges: {value_ranges}"
            )

        try:
            sig_min_shapes, sig_max_shapes = _shapes_from_signature(model_signature, max_batch_size)
        except ModelNavigatorException:
            # it is possible that min_shapes and max_shapes are already there
            if self._dataset_profile_complete():
                return
            raise

        if self.dataset_profile.min_shapes is None:
            self.dataset_profile.min_shapes = sig_min_shapes
            LOGGER.info("Generated default dataset profile: min_shapes=%s.", sig_min_shapes)
        if self.dataset_profile.max_shapes is None:
            if not max_batch_size:
                raise ModelNavigatorException("Cannot construct default dataset profile: max_batch_size not provided")
            if max_batch_size < 0:
                raise ModelNavigatorException("Cannot construct default dataset profile: max_batch_size negative")

            self.dataset_profile.max_shapes = sig_max_shapes
            LOGGER.info("Generated default dataset profile: max_shapes=%s.", sig_max_shapes)

    @property
    def min_shapes(self) -> Dict[str, List[int]]:
        return self.dataset_profile.min_shapes

    @property
    def max_shapes(self) -> Dict[str, List[int]]:
        return self.dataset_profile.max_shapes

    @property
    def opt_shapes(self) -> Dict[str, List[int]]:
        if self.dataset_profile.opt_shapes is None:
            LOGGER.info("opt_shapes not provided, using opt_shapes=max_shapes")
            self.dataset_profile.opt_shapes = self.dataset_profile.max_shapes
        return self.dataset_profile.opt_shapes

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from (
            {
                name: _generate_random(dtype, shapes[name], self.dataset_profile.value_ranges[name], rng=self.rng)
                for name, dtype in self.dataset_profile.dtypes.items()
            }
            for shapes in (self.dataset_profile.min_shapes, self.dataset_profile.max_shapes)
        )


class NavPackageDataloader(Dataloader):
    def __init__(self, package: NavPackage, dataset: str):
        self.package = package
        self.dataset = dataset
        if dataset not in package.datasets:
            raise ModelNavigatorException(f"Dataset {dataset} not found in {package}")

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        self.f_iter = iter(self.package.datasets[self.dataset])
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        file = next(self.f_iter)
        with self.package.open(file) as fobj:
            return dict(np.load(fobj))

    @property
    def min_shapes(self) -> Dict[str, List[int]]:
        shape = {}
        for sample in self:
            for inp, val in sample.items():
                if not shape.get(inp):
                    shape[inp] = list(val.shape)
                assert len(shape[inp]) == len(val.shape)
                for i, (s1, s2) in enumerate(zip(shape[inp], val.shape)):
                    shape[inp][i] = min(s1, s2)
        return shape

    @property
    def max_shapes(self) -> Dict[str, List[int]]:
        shape = {}
        for sample in self:
            for inp, val in sample.items():
                if not shape.get(inp):
                    shape[inp] = list(val.shape)
                assert len(shape[inp]) == len(val.shape)
                for i, (s1, s2) in enumerate(zip(shape[inp], val.shape)):
                    shape[inp][i] = max(s1, s2)
        return shape

    @property
    def opt_shapes(self) -> Dict[str, List[int]]:
        return self.max_shapes
