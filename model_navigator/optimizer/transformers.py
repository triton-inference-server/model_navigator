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
import shutil
from pathlib import Path
from typing import Optional

from model_navigator.config import ModelNavigatorBaseConfig
from model_navigator.log import is_root_logger_verbose
from model_navigator.model import InputModel
from model_navigator.model_navigator_exceptions import ModelNavigatorOptimizerException
from model_navigator.optimizer.polygraphy.transformers import onnx2trt
from model_navigator.optimizer.polygraphy.utils import Profiles
from model_navigator.optimizer.pyt.transformers import ts2onnx
from model_navigator.optimizer.tf2onnx.transformers import tf2onnx, tfopt
from model_navigator.optimizer.utils import (
    KEY_VALUE_SEP,
    MODEL_TRANSFORM_SEP,
    PARAMETERS_SEP,
    TRANSFORM_SPEC_SEP,
    extend_model_name,
)

LOGGER = logging.getLogger(__name__)


def get_optimized_models_dir(workspace_path: Path):
    return workspace_path / "optimized"


class BaseModelTransformer(abc.ABC):
    def __init__(
        self,
        parent: Optional["BaseModelTransformer"] = None,
        *,
        config: Optional[ModelNavigatorBaseConfig] = None,
    ) -> None:
        super().__init__()
        self._parent = parent
        self._config = config
        self._transformed: Optional[InputModel] = None
        self._failed = False

    def run(self, src_model: InputModel) -> InputModel:
        if not self._failed and self._transformed is None:
            if self._parent:
                src_model = self._parent.run(src_model)

            if src_model:
                try:
                    self._transformed = self.transform(src_model)
                except ModelNavigatorOptimizerException:
                    self._failed = True

        return self._transformed

    @property
    def config(self):
        return self._config

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        pass

    def get_output_path(self, src_model: InputModel, *, new_suffix: str) -> Path:
        filename = extend_model_name(src_model.path.stem, self.name) + new_suffix
        return self.export_dir / filename

    @property
    def export_dir(self):
        workspace_path = Path(self._config.workspace_path) if self._config else Path.cwd() / "workspace"
        return get_optimized_models_dir(workspace_path)


class CopyModelFilesTransformer(BaseModelTransformer):
    def __init__(
        self,
        parent: Optional[BaseModelTransformer] = None,
        *,
        export_dir: Path,
        config: Optional[ModelNavigatorBaseConfig] = None,
    ) -> None:
        super().__init__(parent, config=config)
        self._export_dir = export_dir

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        def _replace_separators(name):
            # replace existing separators in model name and filename to enable structure
            name = name.replace(MODEL_TRANSFORM_SEP, PARAMETERS_SEP)
            return name.replace(TRANSFORM_SPEC_SEP, PARAMETERS_SEP)

        model_name = _replace_separators(src_model.name)
        filename = _replace_separators(src_model.path.stem) + src_model.path.suffix

        output_path = self._export_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"Ensuring {output_path} removed")
        shutil.rmtree(output_path, ignore_errors=True)
        LOGGER.debug(f"Copy {src_model.path} to {output_path}")
        if src_model.path.is_dir():
            shutil.copytree(src_model.path, output_path)
        else:
            shutil.copy2(src_model.path, output_path)
        return InputModel(name=model_name, path=output_path, config=self._config)

    @property
    def name(self):
        return "copy"


class TorchScriptAnnotationGenerator(BaseModelTransformer):
    def __init__(self, parent: Optional[BaseModelTransformer] = None, *, config: ModelNavigatorBaseConfig) -> None:

        super().__init__(parent, config=config)

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        from model_navigator.tensor import IOSpec

        LOGGER.debug(f"Annotating {src_model.name}")

        inputs = {spec.name: spec for spec in self._config.inputs}
        outputs = {spec.name: spec for spec in self._config.outputs}
        if not inputs:
            LOGGER.error("Missing inputs for TorchScript model. Use `inputs` config to define input specifications.")
            raise ModelNavigatorOptimizerException(
                "Missing inputs for TorchScript model. Use `inputs` config to define input specifications."
            )

        if not outputs:
            LOGGER.error("Missing outputs for TorchScript model. Use `outputs` config to define output specifications.")
            raise ModelNavigatorOptimizerException(
                "Missing outputs for TorchScript model. Use `outputs` config to define output specifications."
            )
        io_spec = IOSpec(inputs=inputs, outputs=outputs)

        LOGGER.debug("inputs:")
        for input_name, spec in io_spec.inputs.items():
            LOGGER.debug(f"\t{input_name}: {spec}")
        LOGGER.debug("outputs:")
        for input_name, spec in io_spec.outputs.items():
            LOGGER.debug(f"\t{input_name}: {spec}")

        annotation_path = src_model.path.parent / f"{src_model.path.stem}.yaml"
        io_spec.write(annotation_path)

        return src_model

    @property
    def name(self) -> str:
        return "annotation"


class RenameModelIOTransformer(BaseModelTransformer):
    def __init__(self, parent: Optional[BaseModelTransformer] = None) -> None:
        super().__init__(parent)

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        model_name = extend_model_name(src_model.name, transform_name=self.name)
        model_path = self.get_output_path(src_model, new_suffix=src_model.path.suffix)
        return InputModel(name=model_name, path=model_path, config=self._config)

    @property
    def name(self) -> str:
        return "renamed"


class ONNX2TRTTransformer(BaseModelTransformer):
    def __init__(self, parent: Optional[BaseModelTransformer] = None, *, config: ModelNavigatorBaseConfig) -> None:
        super().__init__(parent, config=config)
        self._verbose = is_root_logger_verbose()

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        LOGGER.info(f"Running optimization {self.name}")
        input_path = src_model.path
        output_path = self.get_output_path(src_model, new_suffix=".plan")

        # FIXME: why select only first precision?
        precision = self._config.target_precisions[0]
        max_workspace_size = self._config.max_workspace_size

        profiles = None
        if any([shape for shape in [self._config.min_shapes, self._config.opt_shapes, self._config.max_shapes]]):
            profiles = Profiles(
                min_shapes=self._config.min_shapes,
                opt_shapes=self._config.opt_shapes,
                max_shapes=self._config.max_shapes,
            )

        onnx2trt(
            input_path=input_path,
            output_path=output_path,
            precision=precision,
            max_workspace_size=max_workspace_size,
            profiles=profiles,
            rtol=dict(self._config.rtol),
            atol=dict(self._config.atol),
            verbose=self._verbose,
        )

        model_name = extend_model_name(src_model.name, transform_name=self.name)

        return InputModel(name=model_name, path=output_path, config=self._config)

    @property
    def name(self):
        # FIXME: why select only first precision?
        precision = self._config.target_precisions[0]
        parameters = {"": precision.value}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"onnx2trt{PARAMETERS_SEP}{parameters_suffix}"


class TorchScript2ONNXTransformer(BaseModelTransformer):
    def __init__(self, parent: Optional[BaseModelTransformer] = None, *, config: ModelNavigatorBaseConfig) -> None:
        # TODO: move config and verbose into base model transformer
        super().__init__(parent, config=config)
        self._verbose = is_root_logger_verbose()

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        LOGGER.info(f"Running optimization {self.name}")
        input_path = src_model.path
        output_path = self.get_output_path(src_model, new_suffix=".onnx")

        opset = self._config.onnx_opsets[0]
        value_ranges = self._config.value_ranges
        shapes = self._config.max_shapes

        ts2onnx(input_path, opset, output_path, shapes, value_ranges, self._verbose)

        model_name = extend_model_name(src_model.name, transform_name=self.name)
        return InputModel(name=model_name, path=output_path, config=self._config)

    @property
    def name(self) -> str:
        parameters = {"op": self._config.onnx_opsets[0]}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"ts2onnx{PARAMETERS_SEP}{parameters_suffix}"


class TFSavedModel2ONNXTransformer(BaseModelTransformer):
    def __init__(self, parent: Optional[BaseModelTransformer] = None, *, config: ModelNavigatorBaseConfig) -> None:
        super().__init__(parent, config=config)
        self._verbose = is_root_logger_verbose()

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        LOGGER.info(f"Running optimization {self.name}")
        input_path = src_model.path
        output_path = self.get_output_path(src_model, new_suffix=".onnx")
        opset = self._config.onnx_opsets[0]

        tf2onnx(input_path, output_path, opset=opset, verbose=self._verbose)

        model_name = extend_model_name(src_model.name, transform_name=self.name)
        return InputModel(name=model_name, path=output_path, config=self._config)

    @property
    def name(self) -> str:
        parameters = {"op": self._config.onnx_opsets[0]}
        parameters_suffix = PARAMETERS_SEP.join([f"{k}{KEY_VALUE_SEP}{v}" for k, v in parameters.items()])
        return f"tf2onnx{PARAMETERS_SEP}{parameters_suffix}"


class TFSavedModelOptimizationTransformer(BaseModelTransformer):
    name: str = "tfsmopt"

    def __init__(self, parent: Optional["BaseModelTransformer"] = None, *, config: ModelNavigatorBaseConfig) -> None:
        super().__init__(parent, config=config)
        self._verbose = is_root_logger_verbose()

    def transform(self, src_model: InputModel) -> Optional[InputModel]:
        LOGGER.info(f"Running optimization {self.name}")
        input_path = src_model.path
        output_path = self.get_output_path(src_model, new_suffix=".savedmodel")

        tfopt(input_path, output_path, verbose=self._verbose)
        model_name = extend_model_name(src_model.name, transform_name=self.name)
        return InputModel(name=model_name, path=output_path, config=self._config)
