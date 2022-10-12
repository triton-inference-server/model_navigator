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

import numpy
import pytest
import tensorflow
from polygraphy.backend.trt import Profile

from model_navigator.__version__ import __version__
from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.constants import NAV_PACKAGE_FORMAT_VERSION
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2TFTRT
from model_navigator.framework_api.commands.correctness import Correctness
from model_navigator.framework_api.commands.data_dump.samples import (
    DumpInputModelData,
    DumpOutputModelData,
    samples_to_npz,
)
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.status import ModelStatus, NavigatorStatus
from model_navigator.framework_api.utils import Framework, get_default_max_workspace_size
from model_navigator.model import Format
from model_navigator.tensor import TensorSpec
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
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        model_input_dir = workdir / "model_input"
        dump_cmd = DumpInputModelData()

        input_data = next(iter(dataloader))
        np_input = input_data.numpy()
        samples = [{"input__1": np_input}]

        dump_cmd(
            framework=Framework.TF2,
            workdir=workdir,
            model_name=model_name,
            dataloader=dataloader,
            sample_count=1,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata={"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
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
        model_dir = workdir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_input_dir = workdir / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = workdir / "model_output"

        input_data = next(iter(dataloader))
        np_output = model.predict(input_data)
        np_input = input_data.numpy()
        outputs = [{"output__1": np_output}]

        dump_cmd = DumpOutputModelData()
        samples = [{"input__1": np_input}]

        dump_cmd(
            framework=Framework.TF2,
            workdir=workdir,
            model=model,
            model_name=model_name,
            sample_count=1,
            profiling_sample=samples[0],
            conversion_samples=samples,
            correctness_samples=samples,
            input_metadata={"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)},
            output_metadata={"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)},
            batch_dim=None,
        )

        for filepath in model_output_dir.iterdir():
            dumped_samples = _extract_dumped_samples(filepath)
            for dumped, reference in zip(dumped_samples, outputs):
                for name in reference:
                    assert numpy.allclose(dumped[name], reference[name])


def test_tf2_correctness():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        model_dir = workdir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=model_path, overwrite=True)

        input_data = next(iter(dataloader))
        numpy_output = model.predict(input_data)
        numpy_input = input_data.numpy()
        batch_dim = None

        samples_to_npz([{"input__1": numpy_input}], workdir / "model_input" / "correctness", batch_dim=batch_dim)
        samples_to_npz([{"output__1": numpy_output}], workdir / "model_output" / "correctness", batch_dim=batch_dim)

        input_metadata = TensorMetadata({"input__1": TensorSpec("input__1", numpy_input.shape, numpy_input.dtype)})
        output_metadata = TensorMetadata({"output__1": TensorSpec("output__1", numpy_output.shape, numpy_output.dtype)})
        nav_status = NavigatorStatus(
            format_version=NAV_PACKAGE_FORMAT_VERSION,
            model_navigator_version=__version__,
            uuid="1",
            git_info={},
            environment={},
            export_config={
                "model_name": model_name,
                "target_device": "cpu",
            },
            model_status=[
                ModelStatus(
                    format=Format.TORCHSCRIPT,
                    path=Path("tf-savedmodel/model.savedmodel"),
                    runtime_results=[],
                )
            ],
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            trt_profile=Profile(),
        )
        pkg_desc = PackageDescriptor(nav_status, workdir)
        pkg_desc.save_status_file()

        correctness_cmd = Correctness(
            name="test correctness",
            target_format=Format.TF_SAVEDMODEL,
        )

        correctness_cmd(
            workdir=workdir,
            model_name=model_name,
            rtol=0.0,
            atol=0.0,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            target_device="cpu",
            batch_dim=batch_dim,
            verbose=False
        )


def test_tf2_export_savedmodel():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        model_dir = workdir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)

        input_data = next(iter(dataloader))
        np_output = model.predict(input_data)
        np_input = input_data.numpy()

        input_metadata = TensorMetadata({"input__1": TensorSpec("input__1", np_input.shape, np_input.dtype)})
        output_metadata = TensorMetadata({"output__1": TensorSpec("output__1", np_output.shape, np_output.dtype)})

        export_cmd = ExportTF2SavedModel()

        exported_model_path = workdir / export_cmd(
            model=model,
            model_name=model_name,
            workdir=workdir,
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            verbose=False,
        )
        tensorflow.keras.models.load_model(exported_model_path)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="GPU not available.",
)
def test_tf2_convert_tf_trt():
    from model_navigator.framework_api.commands.data_dump.samples import samples_to_npz

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"

        model_dir = workdir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        input_model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=input_model_path, overwrite=True)

        input_data = next(iter(dataloader))
        samples_to_npz([{"input__1": input_data.numpy()}], workdir / "model_input" / "conversion", None)

        convert_cmd = ConvertSavedModel2TFTRT(target_precision=TensorRTPrecision.FP16)

        converted_model_path = workdir / convert_cmd(
            max_workspace_size=get_default_max_workspace_size(),
            minimum_segment_size=3,
            workdir=workdir,
            model_name=model_name,
            verbose=False,
        )

        tensorflow.keras.models.load_model(converted_model_path)
