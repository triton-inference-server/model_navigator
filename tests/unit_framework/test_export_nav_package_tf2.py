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

import pytest
import tensorflow

import model_navigator as nav

from model_navigator.utils.device import get_gpus

# pytype: enable=import-error

CUDA_AVAILABLE = bool(get_gpus(["all"]))


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


dataloader = [
    tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32)
    for _ in range(5)
]

inp = tensorflow.keras.layers.Input((224, 224, 3))
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)

model = tensorflow.keras.Model(inp, model_output)


def test_tf2_export_savedmodel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            target_formats=(nav.Format.TF_SAVEDMODEL,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)

        # Formats not exported
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp16", format=nav.Format.TF_SAVEDMODEL) is False
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp32", format=nav.Format.TF_SAVEDMODEL) is False


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="GPU not available.",
)
def test_tf2_export_tf_trt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            target_formats=(nav.Format.TF_TRT,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp16", format=nav.Format.TF_SAVEDMODEL)
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp32", format=nav.Format.TF_SAVEDMODEL)

        # Formats not exported but present as step for tf-trt
        assert check_model_dir(model_dir=package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)


def test_tf2_export_tf_onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            target_formats=(nav.Format.ONNX,),
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "onnx", format=nav.Format.ONNX)

        # Formats not exported but present as step for onnx
        assert check_model_dir(model_dir=package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)


# TODO: Fix test - tf2 is using all memory
# def test_tf2_export_trt():
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         model_name = "navigator_model"
#
#         workdir = Path(tmp_dir) / "navigator_workdir"
#         package_dir = workdir / f"{model_name}.nav.workspace"
#         status_file = package_dir / "status.yaml"
#         model_input_dir = package_dir / "model_input"
#         model_output_dir = package_dir / "model_output"
#         navigator_log_file = package_dir / "navigator.log"
#
#         nav.tensorflow.export(
#             model=model,
#             dataloader=dataloader,
#             workdir=workdir,
#             model_name=model_name,
#             override_workdir=True,
#             target_formats=(nav.Format.TENSORRT,),
#             opset=13,
#         )
#
#         assert status_file.is_file()
#         assert model_input_dir.is_dir()
#         assert all(
#             [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
#         )
#         assert model_output_dir.is_dir()
#         assert all(
#             [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
#         )
#         assert navigator_log_file.is_file()
#
#         # Output formats
#         assert check_model_dir(model_dir=package_dir / "trt-fp16", format=nav.Format.TENSORRT)
#         assert check_model_dir(model_dir=package_dir / "trt-fp32", format=nav.Format.TENSORRT)
#
#         # Formats not exported but present as step for TRT
#         assert check_model_dir(model_dir=package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)


def test_tf2_export_string_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav.workspace"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            model_name=model_name,
            override_workdir=True,
            target_formats="tf-trt",
            target_precisions="fp32",
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
        assert navigator_log_file.is_file()

        # Output formats
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp16", format=nav.Format.TF_SAVEDMODEL) is False
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp32", format=nav.Format.TF_SAVEDMODEL) is CUDA_AVAILABLE

        # Formats not exported but present as step for tf-trt
        assert check_model_dir(model_dir=package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)
