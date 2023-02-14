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
import torch  # pytype: disable=import-error

from model_navigator.commands.base import CommandStatus
from model_navigator.commands.infer_metadata import InferInputMetadata
from model_navigator.utils.framework import Framework


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
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output


def test_infer_input_metadata_return_success_status_when_dict_dataloader_with_kwargs_used():
    dataloader = [{"x": torch.randn(1), "z": torch.randn(1)} for _ in range(5)]

    class Model(torch.nn.Module):
        def forward(self, x, y):
            def forward(self, x, y=None, z=None):
                if z is not None:
                    return x + z
                raise ValueError

    status = InferInputMetadata().run(
        framework=Framework.TORCH,
        model=Model(),
        dataloader=dataloader,
        _input_names=("input_x", "input_z"),
    )

    assert status.status == CommandStatus.OK
    assert "input_metadata" in status.output
    assert "dataloader_trt_profile" in status.output
    assert "dataloader_max_batch_size" in status.output
