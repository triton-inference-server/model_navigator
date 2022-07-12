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

import dataclasses
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import click

from model_navigator.cli.spec import (
    BatchingConfigCli,
    ComparatorConfigCli,
    ConversionSetConfigCli,
    DatasetProfileConfigCli,
    ModelConfigCli,
    ModelSignatureConfigCli,
    TritonModelInstancesConfigCli,
)
from model_navigator.common.config import BatchingConfig
from model_navigator.constants import MODEL_NAVIGATOR_DIR
from model_navigator.converter import (
    ComparatorConfig,
    ConversionConfig,
    ConversionLaunchMode,
    ConversionResult,
    Converter,
    DatasetProfileConfig,
)
from model_navigator.converter.config import TargetFormatConfigSetIterator, TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.converter.dataloader import NavPackageDataloader, RandomDataloader
from model_navigator.converter.utils import FORMAT2FRAMEWORK
from model_navigator.core import DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE
from model_navigator.exceptions import ModelNavigatorCliException, ModelNavigatorException
from model_navigator.log import init_logger, log_dict
from model_navigator.model import Format, Model, ModelConfig, ModelSignatureConfig
from model_navigator.results import ResultsStore, State
from model_navigator.triton import DeviceKind, TritonModelInstancesConfig
from model_navigator.utils import Workspace
from model_navigator.utils.cli import clean_workspace_if_needed, common_options, options_from_config
from model_navigator.utils.config import BaseConfig, YamlConfigFile
from model_navigator.utils.device import get_available_device_kinds, get_gpus
from model_navigator.utils.docker import DockerBuilder, DockerImage
from model_navigator.utils.environment import EnvironmentStore, get_env
from model_navigator.utils.nav_package import NavPackage
from model_navigator.utils.source import navigator_install_url, navigator_is_editable
from model_navigator.validators import run_command_validators

LOGGER = logging.getLogger("convert")

_RUN_BY_MODEL_NAVIGATOR = "MODEL_NAVIGATOR_RUN_BY"

TRITON_SUPPORTED_FORMATS = [
    Format.TF_TRT,
    Format.TF_SAVEDMODEL,
    Format.ONNX,
    Format.TENSORRT,
    Format.TORCHSCRIPT,
    Format.TORCH_TRT,
]


@dataclasses.dataclass
class ConversionSetConfig(BaseConfig):
    target_formats: List[Format] = dataclasses.field(default_factory=lambda: TRITON_SUPPORTED_FORMATS)

    # ONNX related
    onnx_opsets: List[int] = dataclasses.field(default_factory=lambda: [14])

    # TRT related
    tensorrt_precisions: List[TensorRTPrecision] = dataclasses.field(
        default_factory=lambda: [TensorRTPrecision.FP16, TensorRTPrecision.TF32]
    )
    tensorrt_precisions_mode: TensorRTPrecisionMode = TensorRTPrecisionMode.HIERARCHY
    tensorrt_explicit_precision: bool = False
    tensorrt_sparse_weights: bool = False
    tensorrt_max_workspace_size: int = DEFAULT_TENSORRT_MAX_WORKSPACE_SIZE

    def __iter__(self):
        for target_format in self.target_formats:
            config_set_iterator = TargetFormatConfigSetIterator.for_target_format(target_format, self)
            yield from config_set_iterator

    @classmethod
    def from_single_config(cls, config: ConversionConfig):
        if not config.target_format:
            return cls(
                target_formats=[],
                tensorrt_precisions=[],
                onnx_opsets=[],
                tensorrt_precisions_mode=config.tensorrt_config.precision_mode,
                tensorrt_explicit_precision=config.tensorrt_config.explicit_precision,
                tensorrt_sparse_weights=config.tensorrt_config.sparse_weights,
                tensorrt_max_workspace_size=config.tensorrt_config.max_workspace_size,
            )

        return cls(
            target_formats=[config.target_format],
            onnx_opsets=[config.onnx_opset] if config.onnx_opset else [],
            tensorrt_precisions=[config.tensorrt_config.precision] or [],
            tensorrt_precisions_mode=config.tensorrt_config.precision_mode,
            tensorrt_explicit_precision=config.tensorrt_config.explicit_precision,
            tensorrt_sparse_weights=config.tensorrt_config.sparse_weights,
            tensorrt_max_workspace_size=config.tensorrt_config.max_workspace_size,
        )


def _run_locally(
    *,
    workspace: Workspace,
    override_workspace: bool = False,
    src_model_config: ModelConfig,
    model_signature_config: ModelSignatureConfig,
    conversion_set_config: ConversionSetConfig,
    batching_config: BatchingConfig,
    comparator_config: ComparatorConfig,
    dataset_profile_config: DatasetProfileConfig,
    device_kinds: List[DeviceKind],
    package: Optional[NavPackage],
    random_seed: int,
    verbose: bool = False,
) -> Sequence[ConversionResult]:
    if not os.environ.get(_RUN_BY_MODEL_NAVIGATOR):
        clean_workspace_if_needed(workspace, override_workspace)

    converter = Converter(workspace=workspace, verbose=verbose)
    conversion_results = []
    for conversion_config in conversion_set_config:
        if package:
            dataloader = NavPackageDataloader(package, "conversion", max_batch_size=batching_config.max_batch_size)
        else:
            dataloader = RandomDataloader(
                src_model_config,
                model_signature_config,
                dataset_profile_config,
                max_batch_size=batching_config.max_batch_size,
                random_seed=random_seed,
            )
        results = converter.convert(
            src_model=src_model_config,
            conversion_config=conversion_config,
            signature_config=model_signature_config,
            comparator_config=comparator_config,
            dataloader=dataloader,
            device_kinds=device_kinds,
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
    batching_config: Optional[BatchingConfig] = None,
    comparator_config: Optional[ComparatorConfig] = None,
    dataset_profile_config: Optional[DatasetProfileConfig] = None,
    instances_config: Optional[TritonModelInstancesConfig] = None,
    framework_docker_image: str,
    model_format: Format,
    devices: List[str],
    verbose: bool = False,
    override_conversion_container: bool = False,
    package: Optional[NavPackage],
    random_seed: int,
) -> Sequence[ConversionResult]:
    clean_workspace_if_needed(workspace, override_workspace)

    config_path = workspace.path / "convert.yaml"
    with YamlConfigFile(config_path) as config_file:
        config_file.save_config(src_model_config)
        config_file.save_config(model_signature_config)
        config_file.save_config(conversion_set_config)
        config_file.save_config(batching_config)
        config_file.save_config(comparator_config)
        config_file.save_config(dataset_profile_config)
        config_file.save_config(instances_config)

    framework = FORMAT2FRAMEWORK[model_format]
    from docker.utils import parse_repository_tag

    _, framework_docker_tag = parse_repository_tag(framework_docker_image)
    converter_docker_image = f"model_navigator_converter:{framework_docker_tag}"

    build_args = {
        "FROM_IMAGE_NAME": framework_docker_image,
    }

    if navigator_is_editable():
        dockerfile_path = MODEL_NAVIGATOR_DIR / "model_navigator/converter/Dockerfile.local"
    else:
        dockerfile_path = MODEL_NAVIGATOR_DIR / "model_navigator/converter/Dockerfile.remote"
        install_url = navigator_install_url(framework, extras=["cli"])
        build_args["INSTALL_URL"] = install_url

    LOGGER.debug(f"Base converter image: {framework_docker_image}")
    LOGGER.debug(f"Converter image: {converter_docker_image}")

    conversion_image = DockerImage(converter_docker_image)
    if not conversion_image.exists() or override_conversion_container:
        conversion_image = DockerBuilder().build(
            dockerfile_path=dockerfile_path,
            image_name=converter_docker_image,
            workdir_path=MODEL_NAVIGATOR_DIR,
            build_args=build_args,
        )

    # run docker container
    verbose_flag = "-v" if verbose else ""
    workspace_flags = f"--workspace-path {workspace.path}"
    workspace_flags += " --override-workspace" if override_workspace else ""
    cmd = (
        "bash -c 'model-navigator convert "
        f"--config-path {config_path.as_posix()} "
        f"--launch-mode local "
        f"{verbose_flag} "
        f"{workspace_flags} "
        f"{package.path.as_posix() if package else ''} "
        "'"
    )

    cwd = Path.cwd()

    required_paths = [workspace.path, src_model_config.model_path.parent, cwd]
    if package:
        required_paths.append(package.path)
    required_paths = sorted({p.resolve() for p in required_paths})

    env = {"PYTHONPATH": cwd.resolve().as_posix(), _RUN_BY_MODEL_NAVIGATOR: 1}
    container = conversion_image.run_container(
        devices=devices, workdir_path=cwd, mount_as_volumes=required_paths, environment=env
    )
    from docker.errors import DockerException

    try:
        LOGGER.debug(f"Running cmd: {cmd}")
        container.run_cmd(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    except DockerException as e:
        raise e
    finally:
        LOGGER.debug(f"Killing docker container {container.id[:8]}")
        container.kill()

    results_store = ResultsStore(workspace)
    results = results_store.load("convert_model")

    # update framework_docker_image when run conversion in docker container
    results = [dataclasses.replace(result, framework_docker_image=framework_docker_image) for result in results]
    results_store.dump("convert_model", results)

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
        try:
            shutil.copytree(result_model_path, output_path)
        except NotADirectoryError:
            try:
                shutil.copy(result_model_path, output_path)
            except shutil.SameFileError:
                LOGGER.info("Result model is already at output path, not copying.")
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
    framework_docker_image: Optional[str],
    gpus: Optional[List[str]],
    launch_mode: ConversionLaunchMode = ConversionLaunchMode.DOCKER,
    override_conversion_container: bool = False,
    package: Optional[NavPackage],
    **kwargs,
):
    src_model_config = ModelConfig.from_dict(kwargs)
    src_model_signature_config = ModelSignatureConfig.from_dict(kwargs)
    conversion_set_config = ConversionSetConfig.from_dict(kwargs)
    batching_config = BatchingConfig.from_dict(kwargs)
    comparator_config = ComparatorConfig.from_dict(kwargs)
    dataset_profile_config = DatasetProfileConfig.from_dict(kwargs)
    instances_config = TritonModelInstancesConfig.from_dict(kwargs)

    src_model = Model(
        name=src_model_config.model_name,
        path=src_model_config.model_path,
        explicit_format=src_model_config.model_format,
        signature_if_missing=src_model_signature_config,
    )

    if not src_model.path.exists():
        LOGGER.error(f"No such file or directory {src_model.path}")
        raise click.Abort()

    framework = FORMAT2FRAMEWORK[src_model.format]
    framework_docker_image = framework_docker_image or framework.container_image(container_version)

    workspace = Workspace(workspace_path)

    gpus = get_gpus(gpus)
    device_kinds = get_available_device_kinds(gpus, instances_config)
    if launch_mode == ConversionLaunchMode.DOCKER:
        if DeviceKind.GPU in device_kinds:
            from docker.types import DeviceRequest

            devices = [DeviceRequest(device_ids=[gpus[0]], capabilities=[["gpu"]])]
        else:
            devices = []

        conversion_results = _run_in_docker(
            workspace=workspace,
            override_workspace=override_workspace,
            src_model_config=src_model_config,
            model_signature_config=src_model_signature_config,
            conversion_set_config=conversion_set_config,
            batching_config=batching_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            instances_config=instances_config,
            framework_docker_image=framework_docker_image,
            model_format=src_model.format,
            devices=devices,
            verbose=verbose,
            override_conversion_container=override_conversion_container,
            package=package,
            random_seed=kwargs.get("random_seed"),
        )
    else:
        if verbose:
            log_dict(
                "convert args:",
                {
                    **dataclasses.asdict(src_model_config),
                    **dataclasses.asdict(conversion_set_config),
                    **dataclasses.asdict(batching_config),
                    **dataclasses.asdict(comparator_config),
                    **dataclasses.asdict(src_model_signature_config),
                    **dataclasses.asdict(dataset_profile_config),
                    "workspace_path": workspace_path,
                    "override_workspace": override_workspace,
                    "output_path": output_path,
                    "container_version": container_version,
                    "framework_docker_image": framework_docker_image,
                    "gpus": gpus,
                },
            )

        conversion_results = _run_locally(
            workspace=workspace,
            override_workspace=override_workspace,
            src_model_config=src_model_config,
            model_signature_config=src_model_signature_config,
            conversion_set_config=conversion_set_config,
            batching_config=batching_config,
            comparator_config=comparator_config,
            dataset_profile_config=dataset_profile_config,
            device_kinds=device_kinds,
            verbose=verbose,
            package=package,
            random_seed=kwargs.get("random_seed"),
        )

    environment_info = get_env()

    environment_store = EnvironmentStore(workspace)
    environment_store.dump("convert_model", environment_info)

    results_store = ResultsStore(workspace)
    results_store.dump("convert_model", conversion_results)

    successful_conversion_results = [result for result in conversion_results if result.status.state == State.SUCCEEDED]
    if not successful_conversion_results:
        raise ModelNavigatorException("No successful conversion performed.")
    elif output_path is not None:
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
@options_from_config(BatchingConfig, BatchingConfigCli)
@options_from_config(DatasetProfileConfig, DatasetProfileConfigCli)
@options_from_config(TritonModelInstancesConfig, TritonModelInstancesConfigCli)
@click.pass_context
def convert_cmd(
    ctx,
    *,
    verbose: bool,
    launch_mode: str,
    override_conversion_container: bool,
    **kwargs,
):
    init_logger(verbose=verbose)
    LOGGER.debug(f"Running '{ctx.command_path}' with config_path: {kwargs.get('config_path')}")

    run_command_validators(
        ctx.command.name,
        configuration={
            "verbose": verbose,
            "launch_mode": launch_mode,
            "override_conversion_container": override_conversion_container,
            **kwargs,
        },
    )

    launch_mode = ConversionLaunchMode(launch_mode)

    try:
        return convert(
            verbose=verbose,
            launch_mode=launch_mode,
            override_conversion_container=override_conversion_container,
            **kwargs,
        )
    except ModelNavigatorException as e:
        message = str(e)
        raise ModelNavigatorCliException(message)
