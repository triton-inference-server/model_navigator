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
# pytype: disable=import-error

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy
import tensorflow

import model_navigator as nav


from model_navigator.framework_api.commands.data_dump.samples import DumpInputModelData, DumpOutputModelData

from model_navigator.tensor import TensorSpec

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
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        model_input_dir = package_dir / "model_input"

        dump_cmd = DumpInputModelData()

        input_data = next(iter(dataloader))
        samples = [{"input__1": input_data}]

        dump_cmd(
            framework=nav.Framework.JAX,
            workdir=workdir,
            model_name=model_name,
            dataloader=dataloader,
            sample_count=1,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata={"input__1": TensorSpec("input__1", input_data.shape, input_data.dtype)},
            batch_dim=None,
        )

        for filepath in model_input_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, samples):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])


def test_tf2_dump_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = package_dir / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = package_dir / "model_output"

        input_data = next(iter(dataloader))
        np_output = predict(input_data, params=params)
        outputs = [{"output__1": np_output}]
        dump_cmd = DumpOutputModelData()
        samples = [{"input__1": input_data}]

        dump_cmd(
            framework=nav.Framework.JAX,
            workdir=workdir,
            model=predict,
            model_params=params,
            model_name=model_name,
            sample_count=1,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata={"input__1": TensorSpec("input__1", input_data.shape, input_data.dtype)},
            output_metadata={"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)},
            batch_dim=None,
        )

        for filepath in model_output_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, outputs):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])
