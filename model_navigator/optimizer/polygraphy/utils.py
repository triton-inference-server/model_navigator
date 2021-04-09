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
from typing import List

import attr

from model_navigator.tensor import TensorSpec


def _no_dynamic_shapes(instance, attribute, value: TensorSpec):
    if not all([isinstance(dim, int) and dim > 0 for dim in value.shape]):
        raise ValueError(
            f"Incorrect profile shape - all dimensions should be defined. For input {value.name} got {value.shape}"
        )


@attr.s
class Profiles:
    min_shapes: List[TensorSpec] = attr.ib(
        attr.validators.deep_iterable(
            member_validator=attr.validators.and_([attr.validators.instance_of(TensorSpec), _no_dynamic_shapes]),
            iterable_validator=attr.validators.instance_of(List),
        ),
        kw_only=True,
    )
    opt_shapes: List[TensorSpec] = attr.ib(
        attr.validators.deep_iterable(
            member_validator=attr.validators.and_([attr.validators.instance_of(TensorSpec), _no_dynamic_shapes]),
            iterable_validator=attr.validators.instance_of(List),
        ),
        kw_only=True,
    )
    max_shapes: List[TensorSpec] = attr.ib(
        attr.validators.deep_iterable(
            member_validator=attr.validators.and_([attr.validators.instance_of(TensorSpec), _no_dynamic_shapes]),
            iterable_validator=attr.validators.instance_of(List),
        ),
        kw_only=True,
    )

    @classmethod
    def from_input_specs(cls, input_specs: List[TensorSpec], max_batch_size: int):
        """
        Replace first dimension (assumed batch axis) with:
        - min batch_size=1
        - opt batch_size=max_batch_size
        - max batch_size=max_batch_size

        Do not update other dimensions
        """
        min_shapes_specs = [attr.evolve(spec, shape=(1,) + spec.shape[1:]) for spec in input_specs]
        max_shapes_specs = [attr.evolve(spec, shape=(max_batch_size,) + spec.shape[1:]) for spec in input_specs]
        opt_shapes_specs = max_shapes_specs
        return cls(min_shapes=min_shapes_specs, opt_shapes=opt_shapes_specs, max_shapes=max_shapes_specs)
