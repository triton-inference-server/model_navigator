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
import logging
import traceback
from typing import Iterable, Optional

from model_navigator.converter.config import ComparatorConfig, ConversionConfig
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.pipelines import ConvertCommandsRegistry
from model_navigator.converter.results import ConversionResult
from model_navigator.converter.transformers import BaseConvertCommand, CompositeConvertCommand
from model_navigator.converter.utils import COMMAND_SPEC_SEP
from model_navigator.exceptions import ModelNavigatorConverterCommandException, ModelNavigatorConverterException
from model_navigator.model import ModelConfig, ModelSignatureConfig
from model_navigator.results import State, Status
from model_navigator.utils.workspace import Workspace

LOGGER = logging.getLogger(__name__)

CONVERTED_DIRNAME = "converted"


class ConvertCommandsExecutor:
    def __init__(self, *, workspace: Workspace, verbose: bool = False):
        LOGGER.debug(f"Convert Commands Executor created; workspace={workspace.path}")
        self._cache = {}
        self._verbose = verbose
        self._workspace = workspace
        self._workdir_path = workspace.path / CONVERTED_DIRNAME

    def __call__(self, src_model: ModelConfig, convert_commands: CompositeConvertCommand) -> Iterable[ConversionResult]:

        for convert_command in convert_commands:
            # check if convert command was already executed on given model
            key = (src_model.model_name, src_model.model_path.resolve().as_posix(), convert_command.name)
            result = self._cache.get(key, None)
            # LOGGER.debug(f"cache[{key}]={'<hit> (state=' + result.status.state.value + ')' if result else '<missed>'}")

            if result is None:
                result = convert_command.transform(self, src_model, verbose=self._verbose)
                self._cache[key] = result

            if result.status.state == State.FAILED:
                break

            src_model = ModelConfig(
                model_name=result.output_model.name,
                model_path=result.output_model.path,
                model_format=result.output_model.format,
            )
        else:
            yield result

    def get_output_path(self, model: ModelConfig, command: BaseConvertCommand):
        suffix = command.file_suffix or ""
        return self._workdir_path / f"{model.model_path.stem}{COMMAND_SPEC_SEP}{command.name}{suffix}"


class Converter:
    def __init__(self, *, workspace: Workspace, verbose: bool = False) -> None:
        LOGGER.debug(f"Converter created; workspace={workspace.path}")
        self._registry = ConvertCommandsRegistry()
        self._executor = ConvertCommandsExecutor(workspace=workspace, verbose=verbose)

    def convert(
        self,
        *,
        src_model: ModelConfig,
        conversion_config: ConversionConfig,
        signature_config: Optional[ModelSignatureConfig] = None,
        comparator_config: Optional[ComparatorConfig] = None,
        dataloader: Dataloader,
    ) -> Iterable[ConversionResult]:
        # if not passed - used default comparator values
        if not comparator_config:
            comparator_config = ComparatorConfig()
            LOGGER.debug(f"There was no specified comparator configuration - created default one {comparator_config}")

        try:
            for composite_commands in self._registry.get(
                model_config=src_model,
                conversion_config=conversion_config,
                signature_config=signature_config,
                comparator_config=comparator_config,
                dataloader=dataloader,
            ):
                for composite_command in composite_commands:
                    yield from self._executor(src_model, composite_command)

        except ModelNavigatorConverterException as e:
            message = str(e)
            LOGGER.error(f"Conversion failed due to invalid configuration: {message}")
            result = ConversionResult(
                status=Status(state=State.FAILED, message=message, log_path=e.log_path),
                source_model_config=src_model,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
            )
            yield result
        except ModelNavigatorConverterCommandException as e:
            message = str(e)
            LOGGER.debug(message)
            result = ConversionResult(
                status=Status(state=State.FAILED, message=message, log_path=e.log_path),
                source_model_config=src_model,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
            )
            yield result
        except Exception:
            message = traceback.format_exc()
            LOGGER.debug(f"Encountered exception \n{message}")
            result = ConversionResult(
                status=Status(state=State.FAILED, message=message, log_path=None),
                source_model_config=src_model,
                conversion_config=conversion_config,
                comparator_config=comparator_config,
            )
            yield result
