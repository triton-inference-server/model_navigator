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
import shutil
from pathlib import Path
from typing import List, Optional

from model_navigator.common.config import TensorRTCommonConfig
from model_navigator.converter.config import ComparatorConfig, ConversionConfig, DatasetProfileConfig
from model_navigator.converter.polygraphy.transformers import onnx2trt
from model_navigator.converter.pyt.transformers import ts2onnx
from model_navigator.converter.results import ConversionResult
from model_navigator.converter.tf2onnx.transformers import tf2onnx, tfopt
from model_navigator.converter.tf_trt.transformers import tf2tftrt
from model_navigator.converter.torch_tensorrt import ts2torchtrt
from model_navigator.converter.utils import KEY_VALUE_SEP, PARAMETERS_SEP, extend_model_name
from model_navigator.exceptions import ModelNavigatorConverterException
from model_navigator.model import Model, ModelConfig, ModelSignatureConfig
from model_navigator.results import State, Status
from model_navigator.utils.config import YamlConfigFile
from model_navigator.utils.dataset import get_shapes, get_value_ranges

LOGGER = logging.getLogger(__name__)


def get_optimized_models_dir(workspace_path: Path):
    return workspace_path / "optimized"


class BaseConvertCommand(abc.ABC):
    def __init__(
        self,
        parent: Optional["BaseConvertCommand"] = None,
    ) -> None:
        super().__init__()
        self._parent = parent
        self._transformed: Optional[Model] = None
        self._failed = False

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    def file_suffix(self):
        return None

    @abc.abstractmethod
    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        pass


class CompositeConvertCommand:
    def __init__(self, cmds: List[BaseConvertCommand]):
        self._cmds = cmds

    def __iter__(self):
        yield from self._cmds


class CopyModelFilesCommand(BaseConvertCommand):
    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.info(f"Running command {self.name} on {model.model_path}")

        model_path = model.model_path

        for path_to_copy in self._scan_for_files_to_copy(model_path):  # obtain also supplementary files
            output_path = self._get_output_path(executor, model, path_to_copy)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            LOGGER.debug(f"Ensuring {output_path} removed")
            shutil.rmtree(output_path, ignore_errors=True)

            LOGGER.debug(f"Copy {path_to_copy} to {output_path}")
            if model_path.is_dir():
                shutil.copytree(path_to_copy, output_path)
            else:
                shutil.copy2(path_to_copy, output_path)

        output_path = self._get_output_path(executor, model, model_path)
        return ConversionResult(
            Status(State.SUCCEEDED, "Model copied", None),
            source_model_config=model,
            conversion_config=None,
            comparator_config=None,
            dataset_profile=None,
            output_model=Model(model.model_name, path=output_path, explicit_format=model.model_format),
        )

    def _get_output_path(self, executor, model, path_to_copy):
        output_path = executor.get_output_path(model, self)
        output_dir = output_path.parent
        return output_dir / path_to_copy.name

    @property
    def name(self):
        return "copy"

    def _scan_for_files_to_copy(self, model_path):
        # supplementary files should have same parent dir and filename
        return list(model_path.parent.glob(f"{model_path.name}*"))


class PassTransformer(BaseConvertCommand):
    def __init__(self, parent: Optional[BaseConvertCommand] = None, *, conversion_config: ConversionConfig) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        return ConversionResult(
            Status(State.SUCCEEDED, "Source model", None),
            source_model_config=model,
            conversion_config=self._conversion_config,
            tensorrt_common_config=None,
            comparator_config=None,
            dataset_profile=None,
            output_model=Model(model.model_name, path=model.model_path, explicit_format=model.model_format),
        )

    @property
    def name(self):
        return "pass"


class TorchScriptAnnotationGenerator(BaseConvertCommand):
    def __init__(self, parent: Optional[BaseConvertCommand] = None, *, signature_config: ModelSignatureConfig) -> None:
        super().__init__(parent)
        self._signature_config = signature_config

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.info(f"Running command {self.name} on {model.model_path}")

        annotation_path = model.model_path.parent / f"{model.model_path.name}.yaml"
        if (not self._signature_config.inputs or not self._signature_config.outputs) and annotation_path.exists():
            with YamlConfigFile(annotation_path) as config_file:
                self._signature_config = config_file.load(ModelSignatureConfig)
            LOGGER.info(f"Signature was not passed. Loaded it from already present annotation file: {annotation_path}")

        if not self._signature_config.inputs:
            raise ModelNavigatorConverterException(
                "Missing inputs for TorchScript model. Use `inputs` config to define input specifications."
            )

        if not self._signature_config.outputs:
            raise ModelNavigatorConverterException(
                "Missing outputs for TorchScript model. Use `outputs` config to define output specifications."
            )

        LOGGER.debug("inputs:")
        for input_name, spec in self._signature_config.inputs.items():
            LOGGER.debug(f"\t{input_name}: {spec}")
        LOGGER.debug("outputs:")
        for input_name, spec in self._signature_config.outputs.items():
            LOGGER.debug(f"\t{input_name}: {spec}")

        annotation_path = model.model_path.parent / f"{model.model_path.name}.yaml"
        LOGGER.info(f"Saving annotations to {annotation_path}")
        with YamlConfigFile(annotation_path) as config_file:
            config_file.save_config(self._signature_config)

        return ConversionResult(
            Status(State.SUCCEEDED, "Model annotated", None),
            source_model_config=model,
            output_model=Model(model.model_name, path=model.model_path, explicit_format=model.model_format),
        )

    @property
    def name(self) -> str:
        return "annotation"


class ONNX2TRTCommand(BaseConvertCommand):
    def __init__(
        self,
        parent: Optional[BaseConvertCommand] = None,
        *,
        conversion_config: ConversionConfig,
        tensorrt_common_config: TensorRTCommonConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._tensorrt_common_config = tensorrt_common_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        onnx2trt(
            input_path=model.model_path,
            output_path=output_path,
            log_path=log_path,
            precision=self._conversion_config.tensorrt_precision,
            precision_mode=self._conversion_config.tensorrt_precision_mode,
            explicit_precision=self._conversion_config.tensorrt_explicit_precision,
            tensorrt_sparse_weights=self._conversion_config.tensorrt_sparse_weights,
            tensorrt_strict_types=self._conversion_config.tensorrt_strict_types,
            max_batch_size=self._comparator_config.max_batch_size if self._comparator_config else None,
            max_workspace_size=self._tensorrt_common_config.tensorrt_max_workspace_size,
            profiles=self._dataset_profile,
            rtol=self._comparator_config.rtol,
            atol=self._comparator_config.atol,
            verbose=bool(verbose),
            input_format=model.model_format,
        )

        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "ONNX converted to TensorRT", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def name(self):
        precision = self._conversion_config.tensorrt_precision
        precision_mode = self._conversion_config.tensorrt_precision_mode
        parameters = {"": precision.value, "m": precision_mode.value[0]}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"polygraphyonnx2trt{PARAMETERS_SEP}{parameters_suffix}"

    @property
    def file_suffix(self):
        return ".plan"


class TorchScript2ONNXCommand(BaseConvertCommand):
    def __init__(
        self,
        parent: Optional[BaseConvertCommand] = None,
        *,
        conversion_config: ConversionConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        model_parsed = Model(model.model_name, model.model_path, explicit_format=model.model_format)
        shapes = get_shapes(model_parsed.signature, self._dataset_profile)
        value_ranges = get_value_ranges(model_parsed.signature, self._dataset_profile)

        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        ts2onnx(
            input_path=model.model_path,
            output_path=output_path,
            log_path=log_path,
            opset=self._conversion_config.onnx_opset,
            shapes=shapes,
            value_ranges=value_ranges,
            verbose=verbose,
        )

        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "TorchScript converted to ONNX", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def name(self) -> str:
        parameters = {"op": self._conversion_config.onnx_opset}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"ts2onnx{PARAMETERS_SEP}{parameters_suffix}"

    @property
    def file_suffix(self):
        return ".onnx"


class TorchTensorRTCommand(BaseConvertCommand):
    """Convert a TorchScript module into a Torch module that executes on TensorRT
    or into a tensorrt plan."""

    def __init__(
        self,
        parent: Optional[BaseConvertCommand] = None,
        *,
        conversion_config: ConversionConfig,
        tensorrt_common_config: TensorRTCommonConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
        signature_config: Optional[ModelSignatureConfig] = None,
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._tensorrt_common_config = tensorrt_common_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile
        self._signature_config = signature_config

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        ts2torchtrt(
            input_path=model.model_path,
            output_path=output_path,
            log_path=log_path,
            dataset_profile=self._dataset_profile,
            conversion_config=self._conversion_config,
            signature_config=self._signature_config,
            max_workspace_size=self._tensorrt_common_config.tensorrt_max_workspace_size or 0,
            max_batch_size=self._comparator_config.max_batch_size if self._comparator_config else 0,
            verbose=bool(verbose),
        )

        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "TorchScript converted to TensorRT", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def name(self) -> str:
        parameters = {"precision": self._conversion_config.tensorrt_precision}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"torch_tensorrt_module{PARAMETERS_SEP}{parameters_suffix}"

    @property
    def file_suffix(self):
        return ".pt"


class TFSavedModel2ONNXTransform(BaseConvertCommand):
    def __init__(
        self,
        parent: Optional[BaseConvertCommand] = None,
        *,
        conversion_config: ConversionConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        input_path = model.model_path
        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        tf2onnx(
            input_path,
            output_path,
            log_path=log_path,
            opset=self._conversion_config.onnx_opset,
            verbose=bool(verbose),
        )

        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "TF SavedModel converted to ONNX", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def name(self) -> str:
        parameters = {"op": self._conversion_config.onnx_opset}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"tf2onnx{PARAMETERS_SEP}{parameters_suffix}"

    @property
    def file_suffix(self):
        return ".onnx"


class TFSavedModel2TFTRTTransform(BaseConvertCommand):
    def __init__(
        self,
        parent: Optional[BaseConvertCommand] = None,
        *,
        conversion_config: ConversionConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
        tensorrt_common_config: TensorRTCommonConfig,
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile
        self._tensorrt_common_config = tensorrt_common_config

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        tf2tftrt(
            input_path=model.model_path.as_posix(),
            output_path=output_path,
            log_path=log_path,
            precision=self._conversion_config.tensorrt_precision,
            dataset_profile=self._dataset_profile,
            max_workspace_size=self._tensorrt_common_config.tensorrt_max_workspace_size,
            max_batch_size=self._comparator_config.max_batch_size if self._comparator_config else None,
            verbose=bool(verbose),
        )

        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "TF SavedModel converted to TF-TRT SavedModel", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def name(self) -> str:
        precision = self._conversion_config.tensorrt_precision
        parameters = {"": precision.value}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"tf-trt{PARAMETERS_SEP}{parameters_suffix}"

    @property
    def file_suffix(self):
        return ".savedmodel"


class TFSavedModelOptimizationTransform(BaseConvertCommand):
    name: str = "tfsmopt"

    def __init__(
        self,
        parent: Optional["BaseConvertCommand"] = None,
        *,
        conversion_config: ConversionConfig,
        comparator_config: Optional[ComparatorConfig],
        dataset_profile: Optional[DatasetProfileConfig],
    ) -> None:
        super().__init__(parent)
        self._conversion_config = conversion_config
        self._comparator_config = comparator_config
        self._dataset_profile = dataset_profile

    def transform(self, executor, model: ModelConfig, *, verbose: int = 0) -> ConversionResult:
        LOGGER.debug(f"Running command {self.name} on {model.model_path}")

        input_path = model.model_path
        output_path = executor.get_output_path(model, self)
        log_path = Path(f"{output_path}.log")

        tfopt(
            input_path,
            output_path,
            log_path=log_path,
            verbose=bool(verbose),
        )
        model_name = extend_model_name(model.model_name, transform_name=self.name)
        return ConversionResult(
            Status(State.SUCCEEDED, "TF SavedModel optimized", log_path.as_posix()),
            source_model_config=model,
            conversion_config=self._conversion_config,
            comparator_config=self._comparator_config,
            dataset_profile=self._dataset_profile,
            output_model=Model(model_name, path=output_path),
        )

    @property
    def file_suffix(self):
        return ".savedmodel"
