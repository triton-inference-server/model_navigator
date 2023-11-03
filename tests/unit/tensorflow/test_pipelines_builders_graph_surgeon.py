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

from model_navigator.api.config import DeviceKind, Format, OptimizationProfile
from model_navigator.commands.optimize.graph_surgeon import GraphSurgeonOptimize
from model_navigator.configuration.common_config import CommonConfig
from model_navigator.configuration.model.model_config import ONNXConfig
from model_navigator.frameworks import Framework
from model_navigator.pipelines.builders.tensorflow import tensorflow_conversion_builder


def test_tensorflow_conversion_builder_return_graph_surgeon_optimization_when_enabled():
    config = CommonConfig(
        framework=Framework.TENSORFLOW,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.ONNX,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [
            ONNXConfig(
                opset=17,
                dynamic_axes={},
                dynamo_export=False,
                graph_surgeon_optimization=True,
            )
        ],
    }
    pipeline = tensorflow_conversion_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 2
    assert pipeline.execution_units[-1].command == GraphSurgeonOptimize


def test_tensorflow_conversion_builder_does_not_return_graph_surgeon_optimization_when_disabled():
    config = CommonConfig(
        framework=Framework.TENSORFLOW,
        dataloader=[{"input_name": [idx]} for idx in range(10)],
        model=None,
        optimization_profile=OptimizationProfile(),
        runner_names=(),
        sample_count=10,
        target_formats=(Format.ONNX,),
        target_device=DeviceKind.CUDA,
    )

    models_config = {
        Format.ONNX: [
            ONNXConfig(
                opset=17,
                dynamic_axes={},
                dynamo_export=False,
                graph_surgeon_optimization=False,
            )
        ],
    }
    pipeline = tensorflow_conversion_builder(config=config, models_config=models_config)
    assert len(pipeline.execution_units) == 1
