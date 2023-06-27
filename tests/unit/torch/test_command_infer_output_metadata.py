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
import pathlib
import tempfile

import numpy as np
import pytest
import torch  # pytype: disable=import-error

from model_navigator.commands.base import CommandStatus
from model_navigator.commands.data_dump.samples import samples_to_npz
from model_navigator.commands.infer_metadata import InferOutputMetadata
from model_navigator.core.tensor import TensorMetadata
from model_navigator.core.workspace import Workspace
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.frameworks import Framework


def test_infer_output_metadata_return_fails_status_when_invalid_model_used():
    class Model(torch.nn.Module):
        def forward(self, x):
            raise ValueError

    input_metadata = TensorMetadata()
    input_metadata.add(name="input_0", shape=(1,), dtype=np.int32)

    dataloader = [torch.randn(1) for _ in range(5)]

    profiling_sample = [{"input_0": np.random.randint(1, 10, 1, dtype=np.int32)}]

    conversion_samples = []
    for _ in range(5):
        conversion_sample = {"input_0": np.random.randint(1, 10, 1, dtype=np.int32)}
        conversion_samples.append(conversion_sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = pathlib.Path(tmpdir)

        samples_to_npz(samples=profiling_sample, path=workspace / "model_input" / "profiling", batch_dim=0)
        samples_to_npz(samples=conversion_samples, path=workspace / "model_input" / "conversion", batch_dim=0)

        with pytest.raises(ModelNavigatorUserInputError):
            InferOutputMetadata().run(
                framework=Framework.TORCH,
                model=Model(),
                dataloader=dataloader,
                profiling_sample=profiling_sample,
                conversion_samples=conversion_samples,
                input_metadata=input_metadata,
                workspace=Workspace(workspace),
                verbose=True,
            )


def test_infer_output_metadata_return_success_status_when_valid_model_used():
    class Model(torch.nn.Module):
        def forward(self, x):
            return 2 * x

    input_metadata = TensorMetadata()
    input_metadata.add(name="input_0", shape=(1,), dtype=np.int32)

    dataloader = [torch.randn(1) for _ in range(5)]

    profiling_sample = [{"input_0": np.random.randint(1, 10, 1, dtype=np.int32)}]

    conversion_samples = []
    for _ in range(5):
        conversion_sample = {"input_0": np.random.randint(1, 10, 1, dtype=np.int32)}
        conversion_samples.append(conversion_sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = pathlib.Path(tmpdir)

        samples_to_npz(samples=profiling_sample, path=workspace / "model_input" / "profiling", batch_dim=0)
        samples_to_npz(samples=conversion_samples, path=workspace / "model_input" / "conversion", batch_dim=0)

        status = InferOutputMetadata().run(
            framework=Framework.TORCH,
            model=Model(),
            dataloader=dataloader,
            profiling_sample=profiling_sample,
            conversion_samples=conversion_samples,
            input_metadata=input_metadata,
            workspace=Workspace(workspace),
            verbose=True,
        )

    assert status.status == CommandStatus.OK
    assert "output_metadata" in status.output
