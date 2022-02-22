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
# pytype: disable=import-error
import tempfile
from pathlib import Path

import numpy
import tensorflow

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2TFTRT
from model_navigator.framework_api.commands.correctness.tf import CorrectnessSavedModel
from model_navigator.framework_api.commands.data_dump.samples import DumpInputModelData, DumpOutputModelData
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.utils import Format, Framework, get_default_max_workspace_size
from model_navigator.tensor import TensorSpec

# pytype: enable=import-error

VALUE_IN_TENSOR = 9.0


def dataloader():
    yield tensorflow.fill(dims=[1, 224, 224, 3], value=VALUE_IN_TENSOR)


inp = tensorflow.keras.layers.Input((224, 224, 3))
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)

model = tensorflow.keras.Model(inp, model_output)


def test_tf2_dump_model_input():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_input_dir = package_dir / "model_input"
        dump_cmd = DumpInputModelData()

        input_data = next(dataloader())
        np_input = input_data.numpy()

        dump_cmd(
            framework=Framework.TF2,
            workdir=workdir,
            model_name=model_name,
            dataloader=dataloader,
            sample_count=1,
            samples=[{"input__1": np_input}],
            input_metadata={"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
        )

        for sample in [numpy.load(npz_file) for npz_file in model_input_dir.iterdir() if model_input_dir.is_dir()]:
            for dumped, reference in zip([sample[array_name] for array_name in sample.files], [input_data]):
                assert len(dumped) == len(reference)
                assert numpy.allclose(dumped, reference)


def test_tf2_dump_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = package_dir / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = package_dir / "model_output"

        input_data = next(dataloader())
        np_output = model.predict(input_data)
        np_input = input_data.numpy()

        dump_cmd = DumpOutputModelData()

        dump_cmd(
            framework=Framework.TF2,
            workdir=workdir,
            model=model,
            model_name=model_name,
            sample_count=1,
            samples=[{"input__1": np_input}],
            input_metadata={"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
            output_metadata={"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)},
        )

        for sample in [numpy.load(npz_file) for npz_file in model_output_dir.iterdir() if model_output_dir.is_dir()]:
            for dumped, reference in zip([sample[array_name] for array_name in sample.files], [np_output]):
                assert len(dumped) == len(reference)
                assert numpy.allclose(dumped, reference)


def test_tf2_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=model_path, overwrite=True)

        input_data = next(dataloader())
        np_output = model.predict(input_data)
        np_input = input_data.numpy()

        correctness_cmd = CorrectnessSavedModel(target_format=Format.TF_SAVEDMODEL)
        correctness_cmd(
            framework=Framework.TF2,
            model=model,
            model_name=model_name,
            workdir=workdir,
            rtol=0.0,
            atol=0.0,
            samples=[{"input__1": np_input}],
            input_metadata={"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
            output_metadata={"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)},
        )


def test_tf2_export_savedmodel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)

        export_cmd = ExportTF2SavedModel()

        exported_model_path = package_dir / export_cmd(model=model, model_name=model_name, workdir=workdir)
        tensorflow.keras.models.load_model(exported_model_path)


def test_tf2_convert_tf_trt():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        input_model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=input_model_path, overwrite=True)

        input_data = next(dataloader())

        convert_cmd = ConvertSavedModel2TFTRT(target_precision=TensorRTPrecision.FP16)

        converted_model_path = package_dir / convert_cmd(
            max_workspace_size=get_default_max_workspace_size(),
            minimum_segment_size=3,
            workdir=workdir,
            model_name=model_name,
            samples=[input_data],
        )

        tensorflow.keras.models.load_model(converted_model_path)
