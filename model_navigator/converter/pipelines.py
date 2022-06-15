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
import abc
import logging
from typing import Iterator, List, Optional, Sequence

from model_navigator.common.config import BatchingConfig
from model_navigator.converter.config import ComparatorConfig, ConversionConfig
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.transformers import (
    CompositeConvertCommand,
    PassTransformer,
    TFSavedModel2ONNXTransform,
    TFSavedModel2TFTRTTransform,
)
from model_navigator.model import Format, Model, ModelConfig, ModelSignatureConfig
from model_navigator.triton import DeviceKind

LOGGER = logging.getLogger(__name__)


class BaseModelPipeline(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
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
        batching_config: Optional[BatchingConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
    ) -> Sequence[CompositeConvertCommand]:

        from model_navigator.converter.transformers import ONNX2TRTCommand

        commands = []

        if conversion_config.target_format == Format.TF_SAVEDMODEL:
            pass_transform = PassTransformer(conversion_config=conversion_config)
            commands.append(CompositeConvertCommand(cmds=[pass_transform]))
        elif conversion_config.target_format == Format.ONNX:
            tf2onnx_converter = TFSavedModel2ONNXTransform(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(CompositeConvertCommand(cmds=[tf2onnx_converter]))

        elif conversion_config.target_format == Format.TENSORRT and DeviceKind.GPU in device_kinds:
            tf2onnx_converter = TFSavedModel2ONNXTransform(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            onnx2trt_command = ONNX2TRTCommand(
                tf2onnx_converter,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(CompositeConvertCommand(cmds=[tf2onnx_converter, onnx2trt_command]))
        elif conversion_config.target_format == Format.TF_TRT and DeviceKind.GPU in device_kinds:
            tf2trt_cmd = TFSavedModel2TFTRTTransform(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(CompositeConvertCommand(cmds=[tf2trt_cmd]))

        return commands


class TorchScriptPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TORCHSCRIPT]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        batching_config: Optional[BatchingConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
    ) -> Sequence[CompositeConvertCommand]:
        from model_navigator.converter.transformers import (
            CopyModelFilesCommand,
            ONNX2TRTCommand,
            TorchScript2ONNXCommand,
            TorchScriptAnnotationGenerator,
            TorchTensorRTCommand,
        )

        commands = []

        if conversion_config.target_format == Format.TORCHSCRIPT:
            pass_transform = PassTransformer(conversion_config=conversion_config)
            commands.append(CompositeConvertCommand(cmds=[pass_transform]))
        elif conversion_config.target_format == Format.ONNX:
            copy_command = CopyModelFilesCommand()
            annotation_command = TorchScriptAnnotationGenerator(copy_command, signature_config=signature_config)
            ts2onnx_converter = TorchScript2ONNXCommand(
                annotation_command,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(CompositeConvertCommand(cmds=[copy_command, annotation_command, ts2onnx_converter]))

        elif conversion_config.target_format == Format.TENSORRT and DeviceKind.GPU in device_kinds:
            copy_command = CopyModelFilesCommand()
            annotation_command = TorchScriptAnnotationGenerator(copy_command, signature_config=signature_config)
            ts2onnx_converter = TorchScript2ONNXCommand(
                annotation_command,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            onnx2trt_command = ONNX2TRTCommand(
                ts2onnx_converter,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(
                CompositeConvertCommand(cmds=[copy_command, annotation_command, ts2onnx_converter, onnx2trt_command])
            )

        elif conversion_config.target_format == Format.TORCH_TRT and DeviceKind.GPU in device_kinds:
            copy_command = CopyModelFilesCommand()
            # TODO: remove annotation step?
            annotation_command = TorchScriptAnnotationGenerator(copy_command, signature_config=signature_config)
            ts2trt_converter = TorchTensorRTCommand(
                annotation_command,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
                signature_config=signature_config,
            )
            commands.append(CompositeConvertCommand(cmds=[copy_command, annotation_command, ts2trt_converter]))

        return commands


class ONNXPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.ONNX]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
    ) -> Sequence[CompositeConvertCommand]:
        from model_navigator.converter.transformers import ONNX2TRTCommand

        commands = []
        if conversion_config.target_format == Format.ONNX:
            pass_transform = PassTransformer(conversion_config=conversion_config)
            commands.append(CompositeConvertCommand(cmds=[pass_transform]))
        elif conversion_config.target_format == Format.TENSORRT and DeviceKind.GPU in device_kinds:
            cmd = ONNX2TRTCommand(
                conversion_config=conversion_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            )
            commands.append(CompositeConvertCommand(cmds=[cmd]))
        return commands


class TRTPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TENSORRT]

    def get_commands(
        self,
        *,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
    ) -> Sequence[CompositeConvertCommand]:
        commands = []
        if conversion_config.target_format == Format.TENSORRT and DeviceKind.GPU in device_kinds:
            pass_transform = PassTransformer(conversion_config=conversion_config)
            commands.append(CompositeConvertCommand(cmds=[pass_transform]))

        return commands


class ConvertCommandsRegistry:
    def get(
        self,
        *,
        model_config: ModelConfig,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Optional[Dataloader] = None,
        device_kinds: List[DeviceKind],
    ) -> Iterator[Sequence[CompositeConvertCommand]]:
        model = Model(model_config.model_name, model_config.model_path, explicit_format=model_config.model_format)

        for pipeline in self._get_pipelines_for_model(model):
            yield pipeline.get_commands(
                conversion_config=conversion_config,
                signature_config=signature_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
                device_kinds=device_kinds,
            )

    def _get_pipelines_for_model(self, model: Model):
        for pipeline_cls in BaseModelPipeline.__subclasses__():
            if model.format in pipeline_cls.src_formats:
                pipeline = pipeline_cls()
                yield pipeline
