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

# pytype: enable=import-error

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


def check_model_dir(model_dir: Path, format: nav.Format) -> bool:
    if not model_dir.is_dir():
        return False
    if not Path(model_dir / "config.yaml").is_file():
        return False
    if format == nav.Format.ONNX:
        model_path = model_dir / "model.onnx"
    elif format == nav.Format.TENSORRT:
        model_path = model_dir / "model.plan"
    else:
        model_path = model_dir / "model.savedmodel"
    if not Path(model_path).exists():
        return False
    return True


dataloader = [numpy.random.rand(1, 10, 10) for _ in range(5)]
params = numpy.random.rand(1, 10, 10)


def predict(inputs, params):
    outputs = jnp.dot(inputs, params)
    return outputs


def test_jax_save_load_savedmodel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        nav_package_path = Path(tmp_dir) / f"{model_name}.nav"

        load_workdir = Path(tmp_dir) / "load_navigator_workdir"
        status_file = load_workdir / "status.yaml"
        model_input_dir = load_workdir / "model_input"
        model_output_dir = load_workdir / "model_output"
        navigator_log_file = load_workdir / "navigator.log"

        pkg_desc = nav.jax.export(
            model=predict,
            model_params=params,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
            run_profiling=False,
            batch_dim=None,
        )

        nav.save(pkg_desc, nav_package_path)
        nav.load(
            nav_package_path,
            workdir=load_workdir,
            retest_conversions=False,
            run_profiling=False,
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file() is False

        # Output formats
        assert check_model_dir(model_dir=load_workdir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)

        # Formats not exported
        assert check_model_dir(model_dir=load_workdir / "onnx", format=nav.Format.ONNX) is False
        assert check_model_dir(model_dir=load_workdir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=load_workdir / "trt-fp32", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=load_workdir / "tf-trt-fp16", format=nav.Format.TF_TRT) is False
        assert check_model_dir(model_dir=load_workdir / "tf-trt-fp32", format=nav.Format.TF_TRT) is False
