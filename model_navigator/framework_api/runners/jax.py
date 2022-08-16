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

import time
from collections import OrderedDict
from typing import Mapping

import numpy
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata

from model_navigator.framework_api.runners.base import INavigatorRunner


class JAXRunner(INavigatorRunner, BaseRunner):
    """
    Runs inference using JAX.
    """

    def __init__(self, model, model_params, input_metadata, output_names, name=None, forward_kw_names=None):
        """
        Args:
            model (Callable): JAX predict function.
            input_metadata (TensorMetadata): Mapping of input names to their data types and shapes.
            output_names (List[str]):
                    A list of output names of the model. This information is used by the
                    Comparator to determine which outputs to compare.
            model_params: (Any): Model weights passed to predict function.

            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="jax-runner")
        self.model = model
        self.model_params = model_params

        self.input_metadata = TensorMetadata()
        for name, spec in input_metadata.items():
            self.input_metadata.add(name, spec.dtype, spec.shape)
        self.output_names = output_names
        self._forward_kw_names = forward_kw_names

    def get_input_metadata_impl(self):
        return self.input_metadata

    def infer_impl(self, feed_dict):
        inputs = tuple(feed_dict.values())

        start = time.time()
        if self._forward_kw_names is None:
            outputs = self.model(*inputs, params=self.model_params)
        else:
            inputs = dict(zip(self._forward_kw_names, inputs))
            outputs = self.model(**inputs, params=self.model_params)
        end = time.time()

        if self.output_names is None:
            if isinstance(outputs, Mapping):
                self.output_names = outputs.keys()
            else:
                self.output_names = [f"output__{i}" for i in range(len(outputs))]

        if isinstance(outputs, numpy.ndarray):
            outputs = (outputs,)
        if isinstance(outputs, Mapping):
            outputs = outputs.values()
        outputs = [numpy.asarray(output) for output in outputs]
        outputs = tuple(outputs)

        out_dict = OrderedDict()
        for name, output in zip(self.output_names, outputs):
            out_dict[name] = output
        self.inference_time = end - start
        return out_dict
