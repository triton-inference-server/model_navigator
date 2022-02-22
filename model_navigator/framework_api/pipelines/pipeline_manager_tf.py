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

from model_navigator.framework_api.commands.config_gen.config_cli import ConfigCli
from model_navigator.framework_api.commands.convert.tf import ConvertSavedModel2TFTRT
from model_navigator.framework_api.commands.correctness.tf import CorrectnessSavedModel
from model_navigator.framework_api.commands.data_dump.samples import (
    DumpInputModelData,
    DumpOutputModelData,
    FetchInputModelData,
)
from model_navigator.framework_api.commands.export.tf import ExportTF2SavedModel
from model_navigator.framework_api.commands.infer_metadata import InferInputMetadata, InferOutputMetadata
from model_navigator.framework_api.commands.performance.tf import PerformanceSavedModel
from model_navigator.framework_api.pipelines.pipeline import Pipeline
from model_navigator.framework_api.pipelines.pipeline_manager_base import PipelineManager
from model_navigator.framework_api.utils import Framework
from model_navigator.model import Format


class TFPipelineManager(PipelineManager):
    def _get_pipeline(self, config) -> Pipeline:
        commands = [
            InferInputMetadata(),
            FetchInputModelData(),
            InferOutputMetadata(),
            ExportTF2SavedModel(),
            CorrectnessSavedModel(target_format=Format.TF_SAVEDMODEL),
            PerformanceSavedModel(target_format=Format.TF_SAVEDMODEL),
            ConfigCli(target_format=Format.TF_SAVEDMODEL),
        ]

        if Format.TF_TRT in config.target_formats:
            for target_precision in config.target_precisions:
                commands.append(ConvertSavedModel2TFTRT(target_precision=target_precision))
                commands.append(CorrectnessSavedModel(target_format=Format.TF_TRT, target_precision=target_precision))
                commands.append(PerformanceSavedModel(target_format=Format.TF_TRT, target_precision=target_precision))
                commands.append(ConfigCli(target_format=Format.TF_TRT, target_precision=target_precision))

        if config.save_data:
            commands.append(DumpInputModelData())
            commands.append(DumpOutputModelData())
        return Pipeline(name="TensorFlow 2 pipeline", framework=Framework.TF2, commands=commands)
