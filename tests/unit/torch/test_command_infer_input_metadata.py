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
import re

import pytest
import torch  # pytype: disable=import-error

from model_navigator import OptimizationProfile
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.infer_metadata import InferInputMetadata
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework


def test_infer_input_metadata_return_success_status_when_invalid_model_used():
    class Model(torch.nn.Module):
        def forward(self, x):
            raise ValueError

    dataloader = [torch.randn(1) for _ in range(5)]

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model,
        dataloader=dataloader,
        _input_names=("input_0",),
        optimization_profile=OptimizationProfile(),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_return_success_status_when_tensor_dataloader_used():
    class Model(torch.nn.Module):
        def forward(self, x):
            return 2 * x

    dataloader = [torch.randn(1) for _ in range(5)]

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model,
        dataloader=dataloader,
        _input_names=("input_0",),
        optimization_profile=OptimizationProfile(),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_return_success_status_when_sequence_dataloader_used():
    dataloader = [[torch.randn(1), torch.randn(1)] for _ in range(5)]

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model(),
        dataloader=dataloader,
        _input_names=("input_0", "input_1"),
        optimization_profile=OptimizationProfile(),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_return_success_status_when_dict_dataloader_used():
    dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model(),
        dataloader=dataloader,
        _input_names=("input_x", "input_z"),
        optimization_profile=OptimizationProfile(),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_return_success_status_when_dict_dataloader_with_kwargs_used():
    dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model(),
        dataloader=dataloader,
        _input_names=("input_x", "input_z"),
        optimization_profile=OptimizationProfile(),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_is_too_big_1dim():
    dataloader = [{"x": torch.randn(idx), "z": torch.randn(idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(6), "z": torch.randn(6)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = re.escape(
        """Provided performance dataloader has invalid shape against the dataset dataloader."""
        """ Performance dataloader shape for input `input_x` is min: (6,), max: (6,)."""
        """ Dataset dataloader shape for input `input_x` is min: (2,), max: (4,)."""
    )

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_is_too_big_2dim():
    dataloader = [{"x": torch.randn(idx, idx), "z": torch.randn(idx, idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(4, 6), "z": torch.randn(4, 6)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = re.escape(
        """Provided performance dataloader has invalid shape against the dataset dataloader."""
        """ Performance dataloader shape for input `input_x` is min: (4, 6), max: (4, 6)."""
        """ Dataset dataloader shape for input `input_x` is min: (2, 2), max: (4, 4)."""
    )

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_is_too_small_1dim():
    dataloader = [{"x": torch.randn(idx), "z": torch.randn(idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(1), "z": torch.randn(2)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = re.escape(
        """Provided performance dataloader has invalid shape against the dataset dataloader."""
        """ Performance dataloader shape for input `input_x` is min: (1,), max: (1,)."""
        """ Dataset dataloader shape for input `input_x` is min: (2,), max: (4,)."""
    )

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_is_too_small_2dim():
    dataloader = [{"x": torch.randn(idx, idx), "z": torch.randn(idx, idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(2, 1), "z": torch.randn(2, 2)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = re.escape(
        """Provided performance dataloader has invalid shape against the dataset dataloader."""
        """ Performance dataloader shape for input `input_x` is min: (2, 1), max: (2, 1)."""
        """ Dataset dataloader shape for input `input_x` is min: (2, 2), max: (4, 4)."""
    )

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_not_match():
    dataloader = [{"x": torch.randn(idx, idx), "z": torch.randn(idx, idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(2), "z": torch.randn(2)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = """Provided performance dataloader does not match dataset dataloader size."""

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_datatype_not_match():
    dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]
    performance_dataloader = [{"x": torch.randint(1, 5, (1,)), "z": torch.randint(1, 5, (1,))} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    match_text = """Provided performance dataloader does not match dataset dataloader data types."""

    with pytest.raises(ModelNavigatorUserInputError, match=match_text):
        InferInputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            _input_names=("input_x", "input_z"),
            optimization_profile=optimization_profile,
        )


def test_infer_input_metadata_raise_error_when_performance_dataloader_shape_is_in_boundaries():
    dataloader = [{"x": torch.randn(idx), "z": torch.randn(idx)} for idx in range(2, 5)]
    performance_dataloader = [{"x": torch.randn(2), "z": torch.randn(4)} for _ in range(5)]

    optimization_profile = OptimizationProfile(dataloader=performance_dataloader)

    class Model(torch.nn.Module):
        def forward(self, x, y=None, z=None):
            if z is not None:
                return x + z
            raise ValueError

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model(),
        dataloader=dataloader,
        _input_names=("input_x", "input_z"),
        optimization_profile=optimization_profile,
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output
