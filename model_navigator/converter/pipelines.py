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
import abc
import logging
from typing import List, Optional, Sequence

from model_navigator.converter.config import ComparatorConfig, ConversionConfig, DatasetProfileConfig
from model_navigator.converter.transformers import CompositeConvertCommand, TFSavedModel2ONNXTransform
from model_navigator.model import Format, Model, ModelConfig, ModelSignatureConfig

LOGGER = logging.getLogger(__name__)


class BaseModelPipeline(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataset_profile: Optional[DatasetProfileConfig] = None,
    ) -> Sequence[CompositeConvertCommand]:
        """Create new transform tree"""
        return []


class SavedModelPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TF_SAVEDMODEL]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataset_profile: Optional[DatasetProfileConfig] = None,
    ) -> Sequence[CompositeConvertCommand]:

        from model_navigator.converter.transformers import ONNX2TRTCommand

        commands = []

        if conversion_config.target_format == Format.ONNX:
            tf2onnx_converter = TFSavedModel2ONNXTransform(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            commands.append(CompositeConvertCommand(cmds=[tf2onnx_converter]))

        elif conversion_config.target_format == Format.TENSORRT:
            tf2onnx_converter = TFSavedModel2ONNXTransform(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            onnx2trt_command = ONNX2TRTCommand(
                tf2onnx_converter,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            commands.append(CompositeConvertCommand(cmds=[tf2onnx_converter, onnx2trt_command]))

        return commands


class TorchScriptPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TORCHSCRIPT]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataset_profile: Optional[DatasetProfileConfig] = None,
    ) -> Sequence[CompositeConvertCommand]:
        from model_navigator.converter.transformers import (
            CopyModelFilesCommand,
            ONNX2TRTCommand,
            TorchScript2ONNXCommand,
            TorchScriptAnnotationGenerator,
        )

        commands = []

        if conversion_config.target_format == Format.ONNX:
            copy_command = CopyModelFilesCommand()
            annotation_command = TorchScriptAnnotationGenerator(copy_command, signature_config=signature_config)
            ts2onnx_converter = TorchScript2ONNXCommand(
                annotation_command,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            commands.append(CompositeConvertCommand(cmds=[copy_command, annotation_command, ts2onnx_converter]))

        elif conversion_config.target_format == Format.TENSORRT:
            copy_command = CopyModelFilesCommand()
            annotation_command = TorchScriptAnnotationGenerator(copy_command, signature_config=signature_config)
            ts2onnx_converter = TorchScript2ONNXCommand(
                annotation_command,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            onnx2trt_command = ONNX2TRTCommand(
                ts2onnx_converter,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            commands.append(
                CompositeConvertCommand(cmds=[copy_command, annotation_command, ts2onnx_converter, onnx2trt_command])
            )

        return commands


class ONNXPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.ONNX]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataset_profile: Optional[DatasetProfileConfig] = None,
    ) -> Sequence[CompositeConvertCommand]:
        from model_navigator.converter.transformers import ONNX2TRTCommand

        commands = []
        if conversion_config.target_format == Format.TENSORRT:
            cmd = ONNX2TRTCommand(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile,
            )
            commands.append(CompositeConvertCommand(cmds=[cmd]))
        return commands


_FORMAT2PIPELINE = {
    format_: pipeline
    for pipeline in [SavedModelPipeline, TorchScriptPipeline, ONNXPipeline]
    for format_ in pipeline.src_formats
}


class ConvertCommandsRegistry:
    def get(
        self,
        *,
        model: ModelConfig,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataset_profile_config: Optional[DatasetProfileConfig] = None,
    ) -> Sequence[CompositeConvertCommand]:
        src_format = Model(model.model_name, model.model_path, explicit_format=model.model_format).format

        try:
            pipeline = _FORMAT2PIPELINE[src_format]()
            return pipeline.get_commands(
                conversion_config=conversion_config,
                signature_config=signature_config,
                comparator_config=comparator_config,
                dataset_profile=dataset_profile_config,
            )
        except KeyError:
            LOGGER.info(f"Have no optimization pipelines defined for {src_format}")
            return []
