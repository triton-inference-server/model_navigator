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
from typing import Iterable, List, Optional, Sequence

import abc
import copy
import logging
import sys
from pathlib import Path

from model_navigator import Format
from model_navigator.config import ModelNavigatorBaseConfig
from model_navigator.framework import PyTorch, TensorFlow2
from model_navigator.model import InputModel
from model_navigator.optimizer.config import (
    TRITON_SUPPORTED_FORMATS,
    OptimizerConfig,
)
from model_navigator.optimizer.runners import PipelineRunInDocker
from model_navigator.optimizer.transformers import (
    BaseModelTransformer,
    TFSavedModel2ONNXTransformer,
    TFSavedModelOptimizationTransformer,
    get_optimized_models_dir,
)

LOGGER = logging.getLogger(__name__)


class BaseModelPipeline(metaclass=abc.ABCMeta):
    def __init__(self, workspace_dir: Optional[Path] = None):
        self._workspace_dir = workspace_dir or Path.cwd() / "workspace"

    @abc.abstractmethod
    def get_transformers(self, config: OptimizerConfig) -> Sequence[BaseModelTransformer]:
        """Create new transform tree"""
        return []

    def execute(self, src_model: InputModel, config: OptimizerConfig) -> Iterable[InputModel]:
        # obtain fresh transform tree and iterate over its leaves
        for transformer in self.get_transformers(config=config):
            # return only not None results of transform tree leaves
            model = transformer.run(src_model)
            if model:
                yield model

    @property
    def export_dir(self):
        return get_optimized_models_dir(self._workspace_dir)

    @staticmethod
    def for_model(*, model: InputModel, config: ModelNavigatorBaseConfig, run_in_container: bool = False):
        cls = _FORMAT2PIPELINE[model.format]
        pipeline = cls(workspace_dir=Path(config.workspace_path))
        if run_in_container:
            framework = _FORMAT2FRAMEWORK[model.format]
            pipeline = PipelineRunInDocker(
                pipeline, framework=framework, config=config, logs_writer=sys.stdout, workdir=Path.cwd()
            )
        return pipeline


def _should_use(transformer: BaseModelTransformer, target_formats: Sequence[Format]):
    return (
        transformer.config and transformer.config.target_format and transformer.config.target_format in target_formats
    )


class SavedModelPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TF_SAVEDMODEL]

    def get_transformers(self, config: OptimizerConfig) -> Sequence[BaseModelTransformer]:
        from model_navigator.optimizer.transformers import ONNX2TRTTransformer

        target_formats = [config.target_format] if config.target_format else TRITON_SUPPORTED_FORMATS
        config.target_format = None

        opt_config = copy.deepcopy(config)
        opt_config.target_format = Format.TF_SAVEDMODEL
        graph_optimizer = TFSavedModelOptimizationTransformer(config=opt_config)
        transformers = [graph_optimizer]

        for opset in config.onnx_opsets:
            # user preprocessed model as transformer performs optimization on its own
            onnx_config = copy.deepcopy(config)
            onnx_config.onnx_opsets = [opset]
            onnx_config.target_format = Format.ONNX

            tf2onnx_converter = TFSavedModel2ONNXTransformer(graph_optimizer, config=onnx_config)
            transformers.append(tf2onnx_converter)
            for precision in config.target_precisions:
                trt_config = copy.deepcopy(onnx_config)
                trt_config.target_precisions = [precision]
                trt_config.target_format = Format.TRT

                transformers.append(ONNX2TRTTransformer(tf2onnx_converter, config=trt_config))

        transformers = [transformer for transformer in transformers if _should_use(transformer, target_formats)]
        return transformers


class TorchScriptPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.TS_SCRIPT, Format.TS_TRACE]

    def get_transformers(self, config: OptimizerConfig) -> Sequence[BaseModelTransformer]:
        from model_navigator.optimizer.transformers import (
            CopyModelFilesTransformer,
            ONNX2TRTTransformer,
            TorchScript2ONNXTransformer,
            TorchScriptAnnotationGenerator,
        )

        target_formats = [config.target_format] if config.target_format else TRITON_SUPPORTED_FORMATS
        config.target_format = None

        copy_transformer = CopyModelFilesTransformer(export_dir=self.export_dir, config=config)

        ts_config = copy.deepcopy(config)
        ts_config.target_format = Format.TS_SCRIPT
        annotation_generator = TorchScriptAnnotationGenerator(copy_transformer, config=ts_config)
        preprocessor = annotation_generator

        transformers = [preprocessor]

        for opset in config.onnx_opsets:
            onnx_config = copy.deepcopy(config)
            onnx_config.onnx_opsets = [opset]
            onnx_config.target_format = Format.ONNX
            ts2onnx_converter = TorchScript2ONNXTransformer(preprocessor, config=onnx_config)
            transformers.append(ts2onnx_converter)

            for precision in config.target_precisions:
                trt_config = copy.deepcopy(onnx_config)
                trt_config.target_precisions = [precision]
                trt_config.target_format = Format.TRT
                onnx2trt_converter = ONNX2TRTTransformer(ts2onnx_converter, config=trt_config)
                transformers.append(onnx2trt_converter)

        transformers = [transformer for transformer in transformers if _should_use(transformer, target_formats)]
        return transformers


class ONNXPipeline(BaseModelPipeline):
    src_formats: List[Format] = [Format.ONNX]

    def get_transformers(self, config: OptimizerConfig) -> Sequence[BaseModelTransformer]:
        from model_navigator.optimizer.transformers import ONNX2TRTTransformer

        target_formats = [config.target_format] if config.target_format else TRITON_SUPPORTED_FORMATS
        config.target_format = None

        transformers = []
        # user preprocessed model as transformer performs optimization on its own
        for precision in config.target_precisions:
            trt_config = copy.deepcopy(config)
            trt_config.target_precisions = [precision]
            trt_config.target_format = Format.TRT

            transformers.append(ONNX2TRTTransformer(config=trt_config))

        transformers = [transformer for transformer in transformers if _should_use(transformer, target_formats)]
        return transformers


_FORMAT2FRAMEWORK = {
    Format.TS_SCRIPT: PyTorch,
    Format.TS_TRACE: PyTorch,
    Format.ONNX: PyTorch,
    Format.TF_SAVEDMODEL: TensorFlow2,
}

_FORMAT2PIPELINE = {
    format_: pipeline
    for pipeline in [SavedModelPipeline, TorchScriptPipeline, ONNXPipeline]
    for format_ in pipeline.src_formats
}


class TransformersRegistry:
    def get(self, *, src_format: Format, config: OptimizerConfig) -> Sequence[BaseModelTransformer]:
        """
        Returns transformers tree, where sequence of returned transformers are leaves of tree.

        If no transformers are available for given pair of formats empty list of transformers is returned.

        Parameters
        ----------
        src_format : Format
            Format of source model
        config : OptimizerConfig
            Configuration of transformer including target format

        Returns
        -------
        Sequence[BaseModelTransformer]
            Sequence of transformers being leaves of transformers tree.
            Empty if no transformers meeting requested formats are available.
        """

        try:
            Pipeline: type(BaseModelPipeline) = _FORMAT2PIPELINE[src_format]
            pipeline = Pipeline()
            return pipeline.get_transformers(config=config)
        except KeyError:
            LOGGER.info(f"Have no optimization pipelines defined for {src_format}")
            return []
