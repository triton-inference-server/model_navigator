# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.tensor import TensorSpec

LOGGER = logging.getLogger(__name__)


def get_shapes(model_signature, dataset_profile):
    if not model_signature.has_input_dynamic_axes():
        shapes = {name: spec.shape for name, spec in model_signature.inputs.items()}
        # assume first axis is batch size; use batch_size=1
        shapes = {name: (1,) + shape[1:] for name, shape in shapes.items()}
    else:
        if not dataset_profile or not dataset_profile.max_shapes:
            raise ModelNavigatorException(
                "Missing model input shapes required during conversion of model with dynamic axes. "
                "Use `max_shapes` config to define missing dataset profiles."
            )
        shapes = dataset_profile.max_shapes
    return shapes


def get_value_ranges(model_signature, dataset_profile):
    if not dataset_profile or not dataset_profile.value_ranges:

        def _get_default_value_range(spec: TensorSpec):
            return {"i": (0, 15), "f": (0.0, 1.0)}[spec.dtype.kind]

        value_ranges = {name: _get_default_value_range(spec) for name, spec in model_signature.inputs.items()}

        LOGGER.info(
            "Missing model input value ranges required during conversion. "
            "Use `value_ranges` config to define missing dataset profiles. "
            f"Used default values_ranges: {value_ranges}"
        )
    else:
        value_ranges = dataset_profile.value_ranges
    return value_ranges
