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

import numpy as np
import tensorflow

import model_navigator as nav
from model_navigator.utils.devices import get_gpus

# pytype: enable=import-error

CUDA_AVAILABLE = bool(get_gpus(["all"]))

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


def check_model_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not Path(model_dir / "config.yaml").is_file():
        return False
    if not Path(model_dir / "model.savedmodel").exists():
        return False
    return True


def test_tf2_tensor_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [
            tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32)
            for _ in range(5)
        ]

        inp = tensorflow.keras.layers.Input((224, 224, 3))
        layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
        model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
        model = tensorflow.keras.Model(inp, model_output)

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            input_names=("input_x",),
            run_profiling=False,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=workdir / "tf-savedmodel")


def test_tf2_sequence_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [
            [
                tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32),
                tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32),
            ]
            for _ in range(5)
        ]

        x, y = tensorflow.keras.layers.Input((224, 224, 3)), tensorflow.keras.layers.Input((224, 224, 3))
        layer_output = tensorflow.keras.layers.Lambda(lambda x: x[0] + x[1])([x, y])
        model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
        model = tensorflow.keras.Model([x, y], model_output)

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            input_names=("input_x", "input_y"),
            run_profiling=False,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=workdir / "tf-savedmodel")


def test_tf2_dict_dataloader():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [
            {
                "x": tensorflow.random.uniform(
                    shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32
                ),
                "y": tensorflow.random.uniform(
                    shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32
                ),
            }
            for _ in range(5)
        ]

        input = {
            "x": tensorflow.keras.layers.Input((224, 224, 3), name="x"),
            "y": tensorflow.keras.layers.Input((224, 224, 3), name="y"),
        }
        layer_output = tensorflow.keras.layers.Lambda(lambda x: x["x"] + x["y"])(input)
        model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
        model = tensorflow.keras.Model(input, model_output)

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            input_names=("input_x", "input_y"),
            run_profiling=False,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=workdir / "tf-savedmodel")


def test_tf2_tensor_dataloader_nan_and_inf_in_and_valid_out():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [tensorflow.constant(np.NaN, dtype=tensorflow.dtypes.float32, shape=[1, 224, 224, 3])]

        inp = tensorflow.keras.layers.Input((224, 224, 3))
        layer_output = tensorflow.keras.layers.Lambda(lambda x: x**0)(inp)
        model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
        model = tensorflow.keras.Model(inp, model_output)

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            input_names=("input_x",),
            run_profiling=False,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=workdir / "tf-savedmodel")


def test_tf2_tensor_dataloader_nan_and_inf_inout():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        status_file = workdir / "status.yaml"
        model_input_dir = workdir / "model_input"
        model_output_dir = workdir / "model_output"
        navigator_log_file = workdir / "navigator.log"

        dataloader = [tensorflow.constant(np.NaN, dtype=tensorflow.dtypes.float32, shape=[1, 224, 224, 3])]

        inp = tensorflow.keras.layers.Input((224, 224, 3))
        layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
        model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
        model = tensorflow.keras.Model(inp, model_output)

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            input_names=("input_x",),
            run_profiling=False,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
        )

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert [
            path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()
        ] == []

        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=workdir / "tf-savedmodel") is False
