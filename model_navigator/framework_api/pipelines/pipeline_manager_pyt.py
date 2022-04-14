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
from model_navigator.framework_api.commands.convert.pyt import ConvertTorchScript2TorchTensorRT
from model_navigator.framework_api.commands.correctness.pyt import (
    CorrectnessPYT2ONNX,
    CorrectnessPYT2TorchScript,
    CorrectnessPYT2TRT,
)
from model_navigator.framework_api.commands.data_dump.samples import (
    DumpInputModelData,
    DumpOutputModelData,
    FetchInputModelData,
)
from model_navigator.framework_api.commands.export.pyt import ExportPYT2ONNX, ExportPYT2TorchScript
from model_navigator.framework_api.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.framework_api.commands.load import LoadMetadata, LoadSamples
from model_navigator.framework_api.commands.performance.onnx import PerformanceONNX
from model_navigator.framework_api.commands.performance.pyt import PerformanceTorchScript
from model_navigator.framework_api.commands.performance.trt import PerformanceTRT
from model_navigator.framework_api.common import Format
from model_navigator.framework_api.config import Config
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.pipelines.pipeline_manager_base import PipelineManager
from model_navigator.framework_api.utils import Framework, format2runtimes


class TorchPipelineManager(PipelineManager):
    def _get_pipeline(self, config: Config) -> Pipeline:

        commands, preprocess_req = [], ()
        ts_exports = {}

        if config.from_source:
            infer_input = InferInputMetadata()
            fetch_input = FetchInputModelData(requires=(infer_input,))
            infer_output = InferOutputMetadata(requires=(infer_input, fetch_input))
            commands.extend([infer_input, fetch_input, infer_output])
            commands.append(DumpInputModelData(requires=(infer_input, fetch_input)))
            commands.append(DumpOutputModelData(requires=(fetch_input, infer_output)))
            preprocess_req = (infer_input, fetch_input, infer_output)
        else:
            load_metadata = LoadMetadata()
            load_samples = LoadSamples(requires=(load_metadata,))
            commands.extend([load_metadata, load_samples])
            preprocess_req = (load_metadata, load_samples)

        for ts_format, torch_trt_format in (
            (Format.TORCHSCRIPT_SCRIPT, Format.TORCH_TRT_SCRIPT),
            (Format.TORCHSCRIPT_TRACE, Format.TORCH_TRT_TRACE),
        ):
            if ts_format in config.target_formats or torch_trt_format in config.target_formats:
                export = ExportPYT2TorchScript(target_format=ts_format, requires=preprocess_req)
                commands.append(export)
                commands.append(CorrectnessPYT2TorchScript(target_format=ts_format, requires=(export,)))
                commands.append(PerformanceTorchScript(target_format=ts_format, requires=(export,)))
                commands.append(ConfigCli(target_format=ts_format, requires=(export,)))
                ts_exports[torch_trt_format] = export
        if Format.ONNX in config.target_formats or Format.TENSORRT in config.target_formats:
            onnx_export = ExportPYT2ONNX(requires=preprocess_req)
            commands.append(onnx_export)
            for provider in format2runtimes(Format.ONNX):
                commands.append(CorrectnessPYT2ONNX(runtime_provider=provider, requires=(onnx_export,)))
                commands.append(PerformanceONNX(runtime_provider=provider, requires=(onnx_export,)))
            commands.append(ConfigCli(target_format=Format.ONNX, requires=(onnx_export,)))
        for torch_trt_format in (Format.TORCH_TRT_SCRIPT, Format.TORCH_TRT_TRACE):
            if torch_trt_format in config.target_formats:
                convert = ConvertTorchScript2TorchTensorRT(
                    target_format=torch_trt_format, requires=(ts_exports[torch_trt_format],)
                )
                commands.append(convert)
                commands.append(CorrectnessPYT2TorchScript(target_format=torch_trt_format, requires=(convert,)))
                commands.append(PerformanceTorchScript(target_format=torch_trt_format, requires=(convert,)))
                commands.append(ConfigCli(target_format=torch_trt_format, requires=(convert,)))
        if Format.TENSORRT in config.target_formats:
            for target_precision in config.target_precisions:
                convert = ConvertONNX2TRT(target_precision=target_precision, requires=(onnx_export,))
                commands.append(convert)
                commands.append(CorrectnessPYT2TRT(target_precision=target_precision, requires=(convert,)))
                commands.append(PerformanceTRT(target_precision=target_precision, requires=(convert,)))
                commands.append(
                    ConfigCli(target_format=Format.TENSORRT, target_precision=target_precision, requires=(convert,))
                )

        return Pipeline(name="PyTorch pipeline", framework=Framework.PYT, commands=commands)
