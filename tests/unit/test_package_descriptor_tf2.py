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

import tensorflow

from model_navigator.converter.config import TensorRTPrecision
from model_navigator.framework_api.commands.core import CommandResults, CommandType
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.package_descriptor import PackageDescriptor
from model_navigator.framework_api.pipelines.pipeline import PipelineResults
from model_navigator.framework_api.utils import Format, Framework, Status

# pytype: enable=import-error


def dataloader():
    yield tensorflow.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tensorflow.dtypes.float32),


inp = tensorflow.keras.layers.Input((224, 224, 3))
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tensorflow.keras.layers.Lambda(lambda x: x)(layer_output)

model = tensorflow.keras.Model(inp, model_output)


def test_tf2_package_descriptor():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "navigator_model"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        model_dir = package_dir / "tf-savedmodel"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.savedmodel"
        tensorflow.keras.models.save_model(model=model, filepath=model_path, overwrite=True)

        config = Config(
            framework=Framework.TF2,
            model_name=model_name,
            model=model,
            dataloader=dataloader,
            workdir=workdir,
            override_workdir=True,
            keep_workdir=True,
            target_formats=(Format.TF_SAVEDMODEL,),
            sample_count=1,
            save_data=False,
            disable_git_info=False,
        )

        cmd_export_result = CommandResults(
            name="Mock export command",
            status=Status.OK,
            command_type=CommandType.EXPORT,
            target_format=Format.TF_SAVEDMODEL,
            target_jit_type=None,
            target_precision=None,
            missing_params={},
            output=None,
        )

        cmd_correctness_result = CommandResults(
            name="Mock correctness command",
            status=Status.OK,
            command_type=CommandType.CORRECTNESS,
            target_format=Format.TF_SAVEDMODEL,
            target_jit_type=None,
            target_precision=None,
            missing_params={},
            output=None,
        )

        pipeline_results = [
            PipelineResults(
                name="Mock pipeline",
                id="mock-pipeline",
                framework=Framework.TF2,
                commands_results=[cmd_export_result, cmd_correctness_result],
            )
        ]

        package_desc = PackageDescriptor(pipeline_results, config)

        # Check model status and load model
        assert package_desc.get_status(format=Format.TF_SAVEDMODEL)
        assert package_desc.get_model(format=Format.TF_SAVEDMODEL) is not None

        # These models should be not available:
        assert package_desc.get_status(format=Format.TF_TRT, precision=TensorRTPrecision.FP16) is False
        assert package_desc.get_model(format=Format.TF_TRT, precision=TensorRTPrecision.FP16) is None

        assert package_desc.get_status(format=Format.TF_TRT, precision=TensorRTPrecision.FP32) is False
        assert package_desc.get_model(format=Format.TF_TRT, precision=TensorRTPrecision.FP32) is None

        assert package_desc.get_status(format=Format.ONNX) is False
        assert package_desc.get_model(format=Format.ONNX) is None
