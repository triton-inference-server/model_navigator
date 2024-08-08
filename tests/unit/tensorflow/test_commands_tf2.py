# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

import pathlib
import tempfile

import keras  # pytype: disable=import-error
import numpy
import pytest
import tensorflow  # pytype: disable=import-error
from packaging.version import Version

from model_navigator.commands.base import CommandStatus
from model_navigator.commands.convert.tf import ConvertSavedModel2TFTRT
from model_navigator.commands.correctness import Correctness
from model_navigator.commands.data_dump.samples import samples_to_npz
from model_navigator.commands.export.tf import ExportTF2SavedModel
from model_navigator.configuration import Format, TensorRTPrecision, TensorRTProfile, TensorType
from model_navigator.configuration.constants import DEFAULT_MAX_WORKSPACE_SIZE
from model_navigator.core.tensor import PyTreeMetadata, TensorMetadata, TensorSpec
from model_navigator.core.workspace import Workspace
from model_navigator.runners.tensorflow import TensorFlowSavedModelCUDARunner
from model_navigator.utils.devices import get_gpus

# pytype: enable=import-error


gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

CUDA_AVAILABLE = bool(get_gpus(["all"]))
VALUE_IN_TENSOR = 9.0

dataloader = [tensorflow.fill(dims=[1, 224, 224, 3], value=VALUE_IN_TENSOR) for _ in range(5)]

inp = tensorflow.keras.layers.Input((224, 224, 3), name="input__1")
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)

model = tensorflow.keras.Model(inp, model_output)


def _extract_dumped_samples(filepath: pathlib.Path):
    dumped_samples = []
    for sample_path in filepath.iterdir():
        sample = {}
        with numpy.load(sample_path.as_posix()) as data:
            for k, v in data.items():
                sample[k] = v
        dumped_samples.append(sample)
    return dumped_samples


def test_tf2_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.savedmodel"
        model_relative_path = pathlib.Path("tf-savedmodel") / "model.savedmodel"
        if Version(keras.__version__) < Version("3.0"):
            tensorflow.keras.models.save_model(  # pytype: disable=module-attr
                model=model, filepath=model_path.as_posix(), overwrite=True
            )
        else:
            model.export(filepath=model_path.as_posix())

        input_data = next(iter(dataloader))
        numpy_output = model.predict(input_data)
        numpy_input = input_data.numpy()
        batch_dim = None

        samples_to_npz([{"input__1": numpy_input}], workspace / "model_input" / "correctness", batch_dim=batch_dim)
        samples_to_npz([{"output__1": numpy_output}], workspace / "model_output" / "correctness", batch_dim=batch_dim)

        input_metadata = TensorMetadata({"input__1": TensorSpec("input__1", numpy_input.shape, numpy_input.dtype)})
        output_metadata = TensorMetadata({"output__1": TensorSpec("output__1", numpy_output.shape, numpy_output.dtype)})

        command_output = Correctness().run(
            workspace=Workspace(workspace),
            format=Format.TF_SAVEDMODEL,
            runner_cls=TensorFlowSavedModelCUDARunner,
            path=model_relative_path,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            batch_dim=batch_dim,
            verbose=True,
        )
        assert command_output.status == CommandStatus.OK


def test_tf2_export_savedmodel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_relative_path = pathlib.Path("tf-savedmodel") / "model.savedmodel"
        exported_model_path = workspace / model_relative_path
        model_dir.mkdir(parents=True, exist_ok=True)

        input_data = next(iter(dataloader))
        np_output = model.predict(input_data)
        np_input = input_data.numpy()

        input_metadata = TensorMetadata(
            {"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
            pytree_metadata=PyTreeMetadata("input__1", TensorType.TENSORFLOW),
        )
        output_metadata = TensorMetadata(
            {"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)},
            pytree_metadata=PyTreeMetadata("output__1", TensorType.TENSORFLOW),
        )

        command_output = ExportTF2SavedModel().run(
            model=model,
            path=model_relative_path,
            workspace=Workspace(workspace),
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            verbose=True,
        )
        assert command_output.status == CommandStatus.OK

        if Version(keras.__version__) < Version("3.0"):
            tensorflow.keras.models.load_model(exported_model_path)  # pytype: disable=module-attr
        else:
            tensorflow.saved_model.load(exported_model_path)  # pytype: disable=module-attr


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="GPU not available.",
)
def test_tf2_convert_tf_trt():
    from model_navigator.commands.data_dump.samples import samples_to_npz

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = pathlib.Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_relative_path = pathlib.Path("tf-trt") / "model.savedmodel"
        converted_model_path = workspace / model_relative_path
        model_dir.mkdir(parents=True, exist_ok=True)
        input_model_path = model_dir / "model.savedmodel"
        if Version(keras.__version__) < Version("3.0"):
            tensorflow.keras.models.save_model(  # pytype: disable=module-attr
                model=model, filepath=input_model_path.as_posix(), overwrite=True
            )
        else:
            model.export(filepath=input_model_path.as_posix())

        input_data = next(iter(dataloader))
        input_data_np = input_data.numpy()
        samples_to_npz([{"input__1": input_data_np}], workspace / "model_input" / "conversion", None)

        input_metadata = TensorMetadata()
        input_metadata.add("input__1", input_data_np.shape, input_data_np.dtype)

        command_output = ConvertSavedModel2TFTRT().run(
            max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
            parent_path=input_model_path,
            path=converted_model_path,
            precision=TensorRTPrecision.FP16,
            minimum_segment_size=3,
            workspace=Workspace(workspace),
            verbose=True,
            input_metadata=input_metadata,
            dataloader_trt_profile=TensorRTProfile(),
            custom_args={},
        )

        assert command_output.status == CommandStatus.OK
        if Version(keras.__version__) < Version("3.0"):
            tensorflow.keras.models.load_model(converted_model_path)  # pytype: disable=module-attr
        else:
            tensorflow.saved_model.load(converted_model_path)  # pytype: disable=module-attr
