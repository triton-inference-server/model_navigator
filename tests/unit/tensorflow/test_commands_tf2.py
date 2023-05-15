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

import numpy
import pytest
import tensorflow  # pytype: disable=import-error

from model_navigator.api.config import Format, TensorRTPrecision, TensorRTProfile
from model_navigator.commands.base import CommandStatus
from model_navigator.commands.convert.tf import ConvertSavedModel2TFTRT
from model_navigator.commands.correctness import Correctness
from model_navigator.commands.data_dump.samples import DumpInputModelData, DumpOutputModelData, samples_to_npz
from model_navigator.commands.export.tf import ExportTF2SavedModel
from model_navigator.core.constants import DEFAULT_MAX_WORKSPACE_SIZE
from model_navigator.core.tensor import TensorMetadata, TensorSpec
from model_navigator.frameworks import Framework
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


def _extract_dumped_samples(filepath: Path):
    dumped_samples = []
    for sample_path in filepath.iterdir():
        sample = {}
        with numpy.load(sample_path.as_posix()) as data:
            for k, v in data.items():
                sample[k] = v
        dumped_samples.append(sample)
    return dumped_samples


def test_tf2_dump_model_input():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"
        model_input_dir = workspace / "model_input"

        input_data = next(iter(dataloader))
        np_input = input_data.numpy()
        samples = [{"input__1": np_input}]

        command_output = DumpInputModelData().run(
            workspace=workspace,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            batch_dim=None,
        )

        assert command_output.status == CommandStatus.OK
        for filepath in model_input_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, samples):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])


def test_tf2_dump_model_output():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"
        model_dir = workspace / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = workspace / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = workspace / "model_output"

        input_data = next(iter(dataloader))
        np_output = model.predict(input_data)
        np_input = input_data.numpy()
        outputs = [{"output__1": np_output}]

        samples = [{"input__1": np_input}]

        command_output = DumpOutputModelData().run(
            framework=Framework.TENSORFLOW,
            workspace=workspace,
            model=model,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata=TensorMetadata({"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)}),
            output_metadata=TensorMetadata({"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)}),
            batch_dim=None,
        )

        assert command_output.status == CommandStatus.OK
        for filepath in model_output_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, outputs):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])


def test_tf2_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.savedmodel"
        model_relative_path = Path("tf-savedmodel") / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=model_path, overwrite=True)

        input_data = next(iter(dataloader))
        numpy_output = model.predict(input_data)
        numpy_input = input_data.numpy()
        batch_dim = None

        samples_to_npz([{"input__1": numpy_input}], workspace / "model_input" / "correctness", batch_dim=batch_dim)
        samples_to_npz([{"output__1": numpy_output}], workspace / "model_output" / "correctness", batch_dim=batch_dim)

        input_metadata = TensorMetadata({"input__1": TensorSpec("input__1", numpy_input.shape, numpy_input.dtype)})
        output_metadata = TensorMetadata({"output__1": TensorSpec("output__1", numpy_output.shape, numpy_output.dtype)})

        command_output = Correctness().run(
            workspace=workspace,
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
        workspace = Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_relative_path = Path("tf-savedmodel") / "model.savedmodel"
        exported_model_path = workspace / model_relative_path
        model_dir.mkdir(parents=True, exist_ok=True)

        input_data = next(iter(dataloader))
        np_output = model.predict(input_data)
        np_input = input_data.numpy()

        input_metadata = TensorMetadata({"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)})
        output_metadata = TensorMetadata({"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)})

        command_output = ExportTF2SavedModel().run(
            model=model,
            path=model_relative_path,
            workspace=workspace,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            verbose=True,
        )
        assert command_output.status == CommandStatus.OK
        tensorflow.keras.models.load_model(exported_model_path)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="GPU not available.",
)
def test_tf2_convert_tf_trt():
    from model_navigator.commands.data_dump.samples import samples_to_npz

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir) / "navigator_workspace"

        model_dir = workspace / "tf-savedmodel"
        model_relative_path = Path("tf-trt") / "model.savedmodel"
        converted_model_path = workspace / model_relative_path
        model_dir.mkdir(parents=True, exist_ok=True)
        input_model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=input_model_path, overwrite=True)

        input_data = next(iter(dataloader))
        samples_to_npz([{"input__1": input_data.numpy()}], workspace / "model_input" / "conversion", None)

        command_output = ConvertSavedModel2TFTRT().run(
            max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
            parent_path=input_model_path,
            path=converted_model_path,
            precision=TensorRTPrecision.FP16,
            minimum_segment_size=3,
            workspace=workspace,
            verbose=True,
            dataloader_trt_profile=TensorRTProfile(),
        )

        assert command_output.status == CommandStatus.OK
        tensorflow.keras.models.load_model(converted_model_path)
