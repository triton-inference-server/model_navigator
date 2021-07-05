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

import dataclasses
import itertools
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import click
from docker.errors import DockerException
from docker.types import DeviceRequest

from model_navigator.cli.spec import (
    ComparatorConfigCli,
    ConversionSetConfigCli,
    DatasetProfileConfigCli,
    ModelConfigCli,
    ModelSignatureConfigCli,
)
from model_navigator.constants import MODEL_NAVIGATOR_DIR
from model_navigator.converter import (
    ComparatorConfig,
    ConversionConfig,
    ConversionLaunchMode,
    ConversionResult,
    Converter,
    ConverterContainer,
    DatasetProfileConfig,
)
from model_navigator.converter.config import TensorRTPrecision
from model_navigator.converter.utils import FORMAT2FRAMEWORK
from model_navigator.device.utils import get_gpus
from model_navigator.log import init_logger, log_dict
from model_navigator.model import Format, Model, ModelConfig, ModelSignatureConfig
from model_navigator.results import ResultsStore, State
from model_navigator.utils import Workspace
from model_navigator.utils.cli import clean_workspace_if_needed, common_options, options_from_config
from model_navigator.utils.config import BaseConfig, YamlConfigFile
from model_navigator.utils.docker import DockerBuilder, DockerImage
from model_navigator.utils.source import navigator_install_url, navigator_is_editable

LOGGER = logging.getLogger("convert")

_RUN_BY_MODEL_NAVIGATOR = "MODEL_NAVIGATOR_RUN_BY"

TRITON_SUPPORTED_FORMATS = [Format.TF_SAVEDMODEL, Format.ONNX, Format.TENSORRT, Format.TORCHSCRIPT]


@dataclasses.dataclass
class ConversionSetConfig(BaseConfig):
    target_formats: List[Format] = dataclasses.field(default_factory=lambda: TRITON_SUPPORTED_FORMATS)
    target_precisions: List[TensorRTPrecision] = dataclasses.field(
        default_factory=lambda: [TensorRTPrecision.FP16, TensorRTPrecision.TF32]
    )
    # ONNX related
    onnx_opsets: List[int] = dataclasses.field(default_factory=lambda: [13])
    # TRT related
    max_workspace_size: Optional[int] = None

    def __iter__(self):
        parameters = [self.target_formats, self.onnx_opsets]
        combinations = itertools.product(*parameters)
        # FIXME: this is workaround for now
        for target_format, onnx_opset in combinations:
            if target_format == Format.TENSORRT:
                yield from self._tensorrt_config(onnx_opset)
            else:
                yield from self._conversion_config(target_format, onnx_opset)

    @classmethod
    def from_single_config(cls, config: ConversionConfig):
        if not config.target_format:
            return cls(
                target_formats=[],
                target_precisions=[],
                onnx_opsets=[],
                max_workspace_size=config.max_workspace_size,
            )

        return cls(
            target_formats=[config.target_format],
            target_precisions=[config.target_precision] if config.target_precision else [],
            onnx_opsets=[config.onnx_opset] if config.onnx_opset else [],
            max_workspace_size=config.max_workspace_size,
        )

    def _tensorrt_config(self, onnx_opset):
        for target_precision in self.target_precisions:
            config = ConversionConfig(
                target_format=Format.TENSORRT,
                target_precision=target_precision,
                onnx_opset=onnx_opset,
                max_workspace_size=self.max_workspace_size,
            )
            yield config

    def _conversion_config(self, target_format, onnx_opset):
        config = ConversionConfig(
            target_format=target_format,
            target_precision=None,
            onnx_opset=onnx_opset,
            max_workspace_size=self.max_workspace_size,
        )
        yield config


def _run_locally(
    *,
    workspace: Workspace,
    override_workspace: bool = False,
    src_model_config: ModelConfig,
    model_signature_config: Optional[ModelSignatureConfig] = None,
    conversion_set_config: ConversionSetConfig,
    comparator_config: Optional[ComparatorConfig] = None,
    dataset_profile_config: Optional[DatasetProfileConfig] = None,
    verbose: bool = False,
) -> Sequence[ConversionResult]:
    if not os.environ.get(_RUN_BY_MODEL_NAVIGATOR):
        clean_workspace_if_needed(workspace, override_workspace)

    converter = Converter(workspace=workspace, verbose=verbose)
    conversion_results = []
    for conversion_config in conversion_set_config:
        results = converter.convert(
            src_model=src_model_config,
            conversion_config=conversion_config,
            signature_config=model_signature_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
        )

        results = list(results)
        conversion_results.extend(results)

    return conversion_results


def _run_in_docker(
    *,
    workspace: Workspace,
    override_workspace: bool = False,
    src_model_config: ModelConfig,
    model_signature_config: Optional[ModelSignatureConfig] = None,
    conversion_set_config: ConversionSetConfig,
    comparator_config: Optional[ComparatorConfig] = None,
    dataset_profile_config: Optional[DatasetProfileConfig] = None,
    container_version: str,
    model_format: Format,
    gpus: Optional[List[str]] = None,
    verbose: bool = False,
    override_conversion_container: bool = False,
) -> Sequence[ConversionResult]:
    clean_workspace_if_needed(workspace, override_workspace)

    config_path = workspace.path / "convert.yaml"
    with YamlConfigFile(config_path) as config_file:
        config_file.save_config(src_model_config)
        config_file.save_config(model_signature_config)
        config_file.save_config(conversion_set_config)
        config_file.save_config(comparator_config)
        config_file.save_config(dataset_profile_config)

    framework = FORMAT2FRAMEWORK[model_format]
    base_framework_image_name = ConverterContainer.base_image(
        model_format=model_format, container_version=container_version
    )
    converter_image_name = ConverterContainer.image(model_format=model_format, container_version=container_version)

    build_args = {
        "FROM_IMAGE_NAME": base_framework_image_name,
    }

    if navigator_is_editable():
        dockerfile_path = MODEL_NAVIGATOR_DIR / "model_navigator/converter/Dockerfile.local"
    else:
        dockerfile_path = MODEL_NAVIGATOR_DIR / "model_navigator/converter/Dockerfile.remote"
        install_url = navigator_install_url(framework)
        build_args["INSTALL_URL"] = install_url

    LOGGER.debug(f"Base converter image: {base_framework_image_name}")
    LOGGER.debug(f"Converter image: {converter_image_name}")

    conversion_image = DockerImage(converter_image_name)
    if not conversion_image.exists() or override_conversion_container:
        conversion_image = DockerBuilder().build(
            dockerfile_path=dockerfile_path,
            image_name=converter_image_name,
            workdir_path=MODEL_NAVIGATOR_DIR,
            build_args=build_args,
        )

    # run docker container
    verbose_flag = "-v" if verbose else ""
    workspace_flags = f"--workspace-path {workspace.path}"
    workspace_flags += " --override-workspace" if override_workspace else ""
    cmd = (
        "bash -c 'model-navigator convert "
        f"--config-path {config_path} "
        f"--launch-mode local "
        f"{verbose_flag} "
        f"{workspace_flags}'"
    )
    gpus = get_gpus(gpus)
    devices = [DeviceRequest(device_ids=[gpus[0]], capabilities=[["gpu"]])]
    cwd = Path.cwd()
    mount_as_volumes = [workspace.path, src_model_config.model_path.parent, cwd]
    env = {"PYTHONPATH": cwd.as_posix(), _RUN_BY_MODEL_NAVIGATOR: 1}
    container = conversion_image.run_container(
        devices=devices, workdir_path=cwd, mount_as_volumes=mount_as_volumes, environment=env
    )

    try:
        LOGGER.debug(f"Running cmd: {cmd}")
        container.run_cmd(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    except DockerException as e:
        raise e
    finally:
        # self._fix_mounted_dirs_ownerships(container, mount_as_volumes)
        LOGGER.debug(f"Killing docker container {container.id[:8]}")
        container.kill()

    results_store = ResultsStore(workspace)
    results = results_store.load("convert", ConversionResult)

    return results


def _copy_to_output_path(conversion_results: Sequence[ConversionResult], output_path):
    output_path = Path(output_path)

    successful_conversion_results = [r for r in conversion_results if r.status.state == State.SUCCEEDED]

    result_to_copy = None
    if not successful_conversion_results:
        LOGGER.warning("Obtained no successful conversion results for given model and conversion parameters")
    elif len(successful_conversion_results) > 1:
        msg = f"Obtained more than 1 successful conversion result - copy just first one into {output_path}."
        LOGGER.warning(msg)
        result_to_copy = successful_conversion_results[0]
    else:
        result_to_copy = successful_conversion_results[0]

    if result_to_copy is not None:
        result_model_path = result_to_copy.output_model.path
        LOGGER.debug(f"Copy {result_model_path} to {output_path}")
        shutil.copy(result_model_path, output_path)
        # copy also supplementary files - ex. model io annotation file
        # they have just changed suffix comparing to model path
        for supplementary_file in result_model_path.parent.glob(f"{result_model_path.stem}.*"):
            if supplementary_file == result_model_path:
                continue
            supplementary_file_output_path = output_path.parent / f"{output_path.stem}{supplementary_file.suffix}"
            LOGGER.debug(f"Copy {supplementary_file} to {supplementary_file_output_path}")
            shutil.copy(supplementary_file, supplementary_file_output_path)


# TODO: nargs????


def convert(
    *,
    workspace_path: Path,
    override_workspace: bool,
    verbose: bool,
    output_path: Optional[str],
    container_version: str,
    gpus: Optional[List[str]],
    launch_mode: ConversionLaunchMode = ConversionLaunchMode.DOCKER,
    override_conversion_container: bool = False,
    **kwargs,
):
    src_model_config = ModelConfig.from_dict(kwargs)
    src_model_signature_config = ModelSignatureConfig.from_dict(kwargs)
    conversion_set_config = ConversionSetConfig.from_dict(kwargs)
    comparator_config = ComparatorConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)

    src_model = Model(
        name=src_model_config.model_name,
        path=src_model_config.model_path,
        explicit_format=src_model_config.model_format,
        signature_if_missing=src_model_signature_config,
    )

    if not src_model.path.exists():
        LOGGER.error(f"No such file or directory {src_model.path}")
        raise click.Abort()

    workspace = Workspace(workspace_path)

    if launch_mode == ConversionLaunchMode.DOCKER:
        conversion_results = _run_in_docker(
            workspace=workspace,
            override_workspace=override_workspace,
            src_model_config=src_model_config,
            model_signature_config=src_model_signature_config,
            conversion_set_config=conversion_set_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            container_version=container_version,
            model_format=src_model.format,
            gpus=gpus,
            verbose=verbose,
            override_conversion_container=override_conversion_container,
        )
    else:
        if verbose:
            log_dict(
                "convert args:",
                {
                    **dataclasses.asdict(src_model_config),
                    **dataclasses.asdict(conversion_set_config),
                    **dataclasses.asdict(comparator_config),
                    **dataclasses.asdict(src_model_signature_config),
                    **dataclasses.asdict(dataset_profile_config),
                    "workspace_path": workspace_path,
                    "override_workspace": override_workspace,
                    "output_path": output_path,
                    "container_version": container_version,
                    "gpus": gpus,
                },
            )

        conversion_results = _run_locally(
            workspace=workspace,
            override_workspace=override_workspace,
            src_model_config=src_model_config,
            model_signature_config=src_model_signature_config,
            conversion_set_config=conversion_set_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            verbose=verbose,
        )

    results_store = ResultsStore(workspace)
    results_store.dump("convert", conversion_results)

    if output_path is not None:
        _copy_to_output_path(conversion_results, output_path)

    return conversion_results


@click.command(name="convert", help="Converts models between formats")
@common_options
@options_from_config(ModelConfig, ModelConfigCli)
@click.option("-o", "--output-path", help="Path to the output file.", type=click.Path())
@click.option(
    "--launch-mode",
    type=click.Choice([item.value for item in ConversionLaunchMode]),
    default=ConversionLaunchMode.DOCKER.value,
    help="The method by which to launch conversion. "
    "'local' assume conversion will be run locally. "
    "'docker' build conversion Docker and perform operations inside it.",
)
@click.option(
    "--override-conversion-container", is_flag=True, help="Override conversion container if it already exists."
)
@options_from_config(ModelSignatureConfig, ModelSignatureConfigCli)
@options_from_config(ConversionSetConfig, ConversionSetConfigCli)
@options_from_config(ComparatorConfig, ComparatorConfigCli)
@options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
def convert_cmd(
    *,
    verbose: bool,
    launch_mode: str,
    override_conversion_container: bool,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug("Running convert_cmd")
    launch_mode = ConversionLaunchMode(launch_mode)
    return convert(
        verbose=verbose, launch_mode=launch_mode, override_conversion_container=override_conversion_container, **kwargs
    )
