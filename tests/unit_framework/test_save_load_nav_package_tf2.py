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
import os
import tempfile
from pathlib import Path

import pytest
import tensorflow

import model_navigator as nav
from model_navigator.framework_api.commands.performance import ProfilerConfig
from model_navigator.utils.device import get_gpus

# pytype: enable=import-error


gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

CUDA_AVAILABLE = bool(get_gpus(["all"]))
UNAVAILABLE_FORMATS = () if CUDA_AVAILABLE else ("trt",)
UNAVAILABLE_RUNTIMES = () if CUDA_AVAILABLE else ("tensorrt", "cuda")
MN_TEST_PACKAGES_DIR_VAR_NAME = "MN_TEST_PACKAGES_DIR"


def check_model_dir(model_dir: Path, format: nav.Format, only_config: bool = False) -> bool:
    if not model_dir.is_dir():
        return False
    if not Path(model_dir / "config.yaml").is_file():
        return False
    if only_config:
        return True
    if format == nav.Format.ONNX:
        model_path = model_dir / "model.onnx"
    elif format == nav.Format.TENSORRT:
        model_path = model_dir / "model.plan"
    else:
        model_path = model_dir / "model.savedmodel"
    if not Path(model_path).exists():
        return False
    return True


inp = tensorflow.keras.layers.Input((224, 224, 3))
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model = tensorflow.keras.Model(inp, model_output)

dataloader = [
    tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32)
    for _ in range(10)
]


def test_tf2_save_load_no_retest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        load_workdir = Path(tmp_dir) / "load_navigator_workdir"
        nav_package_path = workdir / f"{model_name}.nav"
        loaded_package_dir = load_workdir / f"{model_name}.nav.workspace"
        status_file = loaded_package_dir / "status.yaml"
        model_input_dir = loaded_package_dir / "model_input"
        model_output_dir = loaded_package_dir / "model_output"
        navigator_log_file = loaded_package_dir / "navigator.log"

        pkg_desc = nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            target_precisions="fp32",
            profiler_config=ProfilerConfig(measurement_interval=100),
        )

        pkg_desc.set_verified(nav.Format.TENSORRT, nav.RuntimeProvider.TRT, precision=nav.TensorRTPrecision.FP32)
        nav.save(pkg_desc, nav_package_path)
        nav.load(
            nav_package_path,
            workdir=load_workdir,
            retest_conversions=False,
            profiler_config=ProfilerConfig(measurement_interval=100),
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

        # Exported formats
        assert check_model_dir(model_dir=loaded_package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)

        # Converted formats
        assert (
            check_model_dir(model_dir=loaded_package_dir / "trt-fp32", format=nav.Format.TENSORRT, only_config=True)
            is CUDA_AVAILABLE
        )
        assert check_model_dir(model_dir=loaded_package_dir / "onnx", format=nav.Format.ONNX, only_config=True)
        assert check_model_dir(
            model_dir=loaded_package_dir / "tf-trt-fp32", format=nav.Format.TF_SAVEDMODEL, only_config=True
        )

        # Formats not exported
        assert check_model_dir(model_dir=loaded_package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=loaded_package_dir / "tf-trt-fp16", format=nav.Format.TF_SAVEDMODEL) is False


def test_tf2_save_load_retest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        load_workdir = Path(tmp_dir) / "load_navigator_workdir"
        nav_package_path = workdir / f"{model_name}.nav"
        loaded_package_dir = load_workdir / f"{model_name}.nav.workspace"
        status_file = loaded_package_dir / "status.yaml"
        model_input_dir = loaded_package_dir / "model_input"
        model_output_dir = loaded_package_dir / "model_output"
        navigator_log_file = loaded_package_dir / "navigator.log"

        pkg_desc = nav.tensorflow.export(
            model=model,
            dataloader=dataloader,
            override_workdir=True,
            workdir=workdir,
            model_name=model_name,
            target_precisions="fp32",
            profiler_config=ProfilerConfig(measurement_interval=100),
        )

        pkg_desc.set_verified(nav.Format.TENSORRT, nav.RuntimeProvider.TRT, precision=nav.TensorRTPrecision.FP32)
        nav.save(pkg_desc, nav_package_path)
        nav.load(
            nav_package_path,
            workdir=load_workdir,
            profiler_config=ProfilerConfig(measurement_interval=100),
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
        assert check_model_dir(model_dir=loaded_package_dir / "tf-savedmodel", format=nav.Format.TF_SAVEDMODEL)
        assert check_model_dir(model_dir=loaded_package_dir / "onnx", format=nav.Format.ONNX)
        assert check_model_dir(model_dir=loaded_package_dir / "trt-fp32", format=nav.Format.TENSORRT) is CUDA_AVAILABLE
        assert check_model_dir(model_dir=loaded_package_dir / "tf-trt-fp32", format=nav.Format.TF_SAVEDMODEL)

        # Formats not exported
        assert check_model_dir(model_dir=loaded_package_dir / "trt-fp16", format=nav.Format.TENSORRT) is False
        assert check_model_dir(model_dir=loaded_package_dir / "tf-trt-fp16", format=nav.Format.TF_SAVEDMODEL) is False


@pytest.mark.skipif(
    MN_TEST_PACKAGES_DIR_VAR_NAME not in os.environ,
    reason=f"{MN_TEST_PACKAGES_DIR_VAR_NAME} variable not found in the environment.",
)
@pytest.mark.parametrize(
    "nav_package_path",
    list((Path(os.environ.get(MN_TEST_PACKAGES_DIR_VAR_NAME)) / "tensorflow").iterdir())
    if MN_TEST_PACKAGES_DIR_VAR_NAME in os.environ
    else (),
)
def test_backward_compatibility(nav_package_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        workdir = Path(tmp_dir) / "navigator_workdir"

        pkg_desc = nav.load(nav_package_path, workdir=workdir, retest_conversions=False)
        old_status = pkg_desc.get_formats_status()
        retest_pkg_desc = nav.load(
            nav_package_path,
            workdir=workdir,
            override_workdir=True,
            retest_conversions=True,
            profiler_config=ProfilerConfig(max_trials=100),
        )
        new_status = retest_pkg_desc.get_formats_status()

        for format, format_status in old_status.items():
            for runtime, runtime_status in format_status.items():
                if any(ft in format.lower() for ft in UNAVAILABLE_FORMATS) or any(
                    rt in runtime.lower() for rt in UNAVAILABLE_RUNTIMES
                ):
                    continue
                assert (
                    runtime_status == new_status[format][runtime]
                ), f"{format} {runtime} status not matching with the old package."
