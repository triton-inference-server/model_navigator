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
"""JAX runner."""

from collections import OrderedDict
from typing import List, Mapping

import jax.numpy as jnp  # pytype: disable=import-error
import numpy

from model_navigator.api.config import Format
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils.dataloader import get_default_output_names


class JAXRunner(NavigatorRunner):
    """Runs inference using JAX."""

    def infer_impl(self, feed_dict):
        """Run inference in JAX.

        Args:
            feed_dict: A dictionary with profiling samples
        """
        inputs = tuple(feed_dict.values())

        if self._input_metadata_mapping is None:
            outputs = self.model(*inputs)
        else:
            inputs = dict(zip(self._input_metadata_mapping, inputs))
            outputs = self.model(**inputs)

        if self.output_metadata:
            output_names = self.output_metadata.keys()
        else:
            output_names = outputs.keys() if isinstance(outputs, Mapping) else get_default_output_names(len(outputs))

        if isinstance(outputs, (numpy.ndarray, jnp.ndarray)):
            outputs = (outputs,)
        if isinstance(outputs, Mapping):
            outputs = outputs.values()
        outputs = [numpy.asarray(output) for output in outputs]
        outputs = tuple(outputs)

        out_dict = OrderedDict()
        for name, output in zip(output_names, outputs):
            out_dict[name] = output
        return out_dict

    @classmethod
    def name(cls) -> str:
        """Return name of the runner."""
        return "JAX"

    @classmethod
    def format(cls) -> Format:
        """Return JAX runner format."""
        return Format.JAX

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]


def register_jax_runners():
    """Register all jax runners."""
    register_runner(JAXRunner)
