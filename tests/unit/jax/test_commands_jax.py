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
# pytype: disable=import-error

import tempfile
from pathlib import Path

import jax.numpy as jnp  # pytype: disable=import-error
import numpy
import tensorflow  # pytype: disable=import-error

from model_navigator.commands.data_dump.samples import DumpInputModelData, DumpOutputModelData
from model_navigator.utils.framework import Framework
from model_navigator.utils.jax import JaxModel
from model_navigator.utils.tensor import TensorMetadata, TensorSpec

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

dataloader = [numpy.random.rand(1, 10, 10) for _ in range(5)]
params = numpy.random.rand(1, 10, 10)


def predict(inputs, params):
    outputs = jnp.dot(inputs, params)
    return outputs


def _extract_dumped_samples(filepath: Path):
    dumped_samples = []
    for sample_path in filepath.iterdir():
        sample = {}
        with numpy.load(sample_path.as_posix()) as data:
            for k, v in data.items():
                sample[k] = v
        dumped_samples.append(sample)
    return dumped_samples


def test_jax_dump_model_input():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"

        model_input_dir = workspace / "model_input"

        input_data = next(iter(dataloader))
        samples = [{"input__1": input_data}]

        DumpInputModelData().run(
            workspace=workspace,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            batch_dim=None,
        )

        for filepath in model_input_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, samples):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])


def test_jax_dump_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"
        model_dir = workspace / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = workspace / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = workspace / "model_output"

        input_data = next(iter(dataloader))
        np_output = predict(input_data, params=params)
        outputs = [{"output__1": np_output}]
        samples = [{"input__1": input_data}]

        DumpOutputModelData().run(
            framework=Framework.JAX,
            workspace=workspace,
            model=JaxModel(predict, params),
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata=TensorMetadata({"input__1": TensorSpec("input__1", input_data.shape, input_data.dtype)}),
            output_metadata=TensorMetadata({"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)}),
            batch_dim=None,
        )

        for filepath in model_output_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, outputs):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])
