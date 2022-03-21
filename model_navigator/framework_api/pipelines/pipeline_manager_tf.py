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
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2ONNX, ConvertSavedModel2TFTRT
from model_navigator.framework_api.commands.correctness.tf import (
    CorrectnessSavedModel,
    CorrectnessTensorFlow2ONNX,
    CorrectnessTensorFlow2TRT,
)
from model_navigator.framework_api.commands.data_dump.samples import (
    DumpInputModelData,
    DumpOutputModelData,
    FetchInputModelData,
)
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.framework_api.commands.performance.onnx import PerformanceONNX
from model_navigator.framework_api.commands.performance.tf import PerformanceSavedModel
from model_navigator.framework_api.commands.performance.trt import PerformanceTRT
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.pipelines.pipeline_manager_base import PipelineManager
from model_navigator.framework_api.utils import Framework, format2runtimes
from model_navigator.model import Format


class TFPipelineManager(PipelineManager):
    def _get_pipeline(self, config) -> Pipeline:
        infer_input = InferInputMetadata()
        fetch_input = FetchInputModelData(requires=(infer_input,))
        infer_output = InferOutputMetadata(requires=(infer_input, fetch_input))
        export_savedmodel = ExportTF2SavedModel(requires=(infer_input, fetch_input, infer_output))

        commands = [
            infer_input,
            fetch_input,
            infer_output,
            export_savedmodel,
            CorrectnessSavedModel(target_format=Format.TF_SAVEDMODEL, requires=(export_savedmodel,)),
            PerformanceSavedModel(target_format=Format.TF_SAVEDMODEL, requires=(export_savedmodel,)),
            ConfigCli(target_format=Format.TF_SAVEDMODEL, requires=(export_savedmodel,)),
        ]

        if Format.ONNX in config.target_formats:
            onnx_convert = ConvertSavedModel2ONNX(requires=(export_savedmodel,))
            commands.append(onnx_convert)
            for provider in format2runtimes(Format.ONNX):
                commands.append(CorrectnessTensorFlow2ONNX(runtime_provider=provider, requires=(onnx_convert,)))
                commands.append(PerformanceONNX(runtime_provider=provider, requires=(onnx_convert,)))
        if Format.TENSORRT in config.target_formats:
            if Format.ONNX not in config.target_formats:
                onnx_convert = ConvertSavedModel2ONNX(requires=(export_savedmodel,))
                commands.append(onnx_convert)
                for provider in format2runtimes(Format.ONNX):
                    commands.append(CorrectnessTensorFlow2ONNX(runtime_provider=provider, requires=(onnx_convert,)))
                    commands.append(PerformanceONNX(runtime_provider=provider, requires=(onnx_convert,)))
            for target_precision in config.target_precisions:
                onnx_convert = ConvertONNX2TRT(target_precision=target_precision, requires=(onnx_convert,))
                commands.append(onnx_convert)
                commands.append(CorrectnessTensorFlow2TRT(target_precision=target_precision, requires=(onnx_convert,)))
                commands.append(PerformanceTRT(target_precision=target_precision, requires=(onnx_convert,)))
                commands.append(
                    ConfigCli(
                        target_format=Format.TENSORRT, target_precision=target_precision, requires=(onnx_convert,)
                    )
                )

        if Format.TF_TRT in config.target_formats:
            for target_precision in config.target_precisions:
                convert_savedmodel2trt = ConvertSavedModel2TFTRT(
                    target_precision=target_precision, requires=(export_savedmodel,)
                )
                commands.extend(
                    [
                        convert_savedmodel2trt,
                        CorrectnessSavedModel(
                            target_format=Format.TF_TRT,
                            target_precision=target_precision,
                            requires=(convert_savedmodel2trt,),
                        ),
                        PerformanceSavedModel(
                            target_format=Format.TF_TRT,
                            target_precision=target_precision,
                            requires=(convert_savedmodel2trt,),
                        ),
                        ConfigCli(
                            target_format=Format.TF_TRT,
                            target_precision=target_precision,
                            requires=(convert_savedmodel2trt,),
                        ),
                    ]
                )

        if config.save_data:
            commands.append(DumpInputModelData(requires=(infer_input, fetch_input)))
            commands.append(DumpOutputModelData(requires=(fetch_input, infer_output)))
        return Pipeline(name="TensorFlow 2 pipeline", framework=Framework.TF2, commands=commands)
