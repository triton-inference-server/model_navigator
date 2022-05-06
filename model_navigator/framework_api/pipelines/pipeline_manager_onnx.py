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

from model_navigator.framework_api.commands.config_gen.config_cli import ConfigCli
from model_navigator.framework_api.commands.convert.onnx import ConvertONNX2TRT
from model_navigator.framework_api.commands.copy.onnx import CopyONNX
from model_navigator.framework_api.commands.correctness.onnx import CorrectnessONNX2TRT
from model_navigator.framework_api.commands.data_dump.samples import (
    DumpInputModelData,
    DumpOutputModelData,
    FetchInputModelData,
)
from model_navigator.framework_api.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.framework_api.commands.load import LoadMetadata, LoadSamples
from model_navigator.framework_api.commands.performance.onnx import PerformanceONNX
from model_navigator.framework_api.commands.performance.trt import PerformanceTRT
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.pipelines.pipeline_manager_base import PipelineManager
from model_navigator.framework_api.utils import Framework
from model_navigator.model import Format


class ONNXPipelineManager(PipelineManager):
    def _get_pipeline(self, config) -> Pipeline:
        commands, preprocess_req = [], ()
        if config.from_source:
            infer_input = InferInputMetadata()
            fetch_input = FetchInputModelData(requires=(infer_input,))
            infer_output = InferOutputMetadata(requires=(infer_input, fetch_input))

            commands.extend(
                [
                    infer_input,
                    fetch_input,
                    infer_output,
                    DumpInputModelData(requires=(infer_input, fetch_input)),
                    DumpOutputModelData(requires=(fetch_input, infer_output)),
                ]
            )
            copy_onnx = CopyONNX()
            commands.append(copy_onnx)
            commands.append(ConfigCli(target_format=Format.ONNX, requires=(copy_onnx,)))
            preprocess_req = (copy_onnx,)
        else:
            load_metadata = LoadMetadata()
            load_samples = LoadSamples(requires=(load_metadata,))
            commands.extend([load_metadata, load_samples])
            preprocess_req = (load_metadata, load_samples)

        for provider in config.onnx_runtimes:
            commands.append(PerformanceONNX(runtime_provider=provider, requires=preprocess_req))

        if Format.TENSORRT in config.target_formats:
            for target_precision in config.target_precisions:
                trt_convert = ConvertONNX2TRT(target_precision=target_precision)
                commands.append(trt_convert)
                commands.append(CorrectnessONNX2TRT(target_precision=target_precision, requires=(trt_convert,)))
                commands.append(PerformanceTRT(target_precision=target_precision, requires=(trt_convert,)))
                commands.append(
                    ConfigCli(target_format=Format.TENSORRT, target_precision=target_precision, requires=(trt_convert,))
                )
        return Pipeline(name="ONNX pipeline", framework=Framework.ONNX, commands=commands)
