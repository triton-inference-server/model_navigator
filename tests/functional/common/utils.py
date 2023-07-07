# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
import os.path
import pathlib
import zipfile
from typing import Dict, List, Union

from model_navigator import CommandStatus
from model_navigator.api.config import EXPORT_FORMATS, INPUT_FORMATS, Format
from model_navigator.commands.performance.performance import Performance
from model_navigator.frameworks import Framework
from model_navigator.package.status import ModelStatus, Status
from model_navigator.pipelines.wrappers.profile import ProfilingResults

FORMAT_FILES = [
    "format.log",
]

EVALUATION_FILES = [
    "reproduce_correctness.py",
    "reproduce_correctness.sh",
    "reproduce_profiling.py",
    "reproduce_profiling.sh",
]

COMMON_FILES = [
    "navigator.log",
    "status.yaml",
    "model_input/correctness/0.npz",
    "model_input/profiling/0.npz",
    "model_output/correctness/0.npz",
    "model_output/profiling/0.npz",
]


class ValidationError(Exception):
    pass


def _conversion_status(current_status: ModelStatus, package_path: pathlib.Path):
    conversion_status = "OK"

    status = current_status.status
    for value in filter(lambda val: val != "OK", status.values()):
        conversion_status = value
        break

    parent_key = current_status.model_config.parent_key
    if not parent_key:
        model_path = current_status.model_config.path
        with zipfile.ZipFile(package_path) as zip_file:
            conversion_status = "FAIL"
            for member in zip_file.namelist():
                if str(member) == str(model_path) or member.startswith(f"{str(model_path)}/"):
                    conversion_status = "OK"

    return conversion_status


def _get_model_status(
    model_status_key: str, models_status: Dict[str, ModelStatus], package_path: pathlib.Path, input_format: Format
):
    current_status = models_status[model_status_key]
    conversion_status = _conversion_status(current_status, package_path)
    if conversion_status != "OK":
        return conversion_status

    parent_conversion_status = None
    parent_key = current_status.model_config.parent_key

    while parent_key is not None or parent_key == input_format:
        current_status = models_status[parent_key]
        parent_conversion_status = _conversion_status(current_status, package_path)
        if parent_conversion_status != "FAIL":
            break

        parent_key = current_status.model_config.parent_key

    if parent_conversion_status == "FAIL":
        return "SKIPPED"
    else:
        return "OK"


def collect_expected_files(package_path: pathlib.Path, status: Status) -> List[str]:
    files = COMMON_FILES
    models_status = status.models_status
    framework = Framework(status.config["framework"])
    input_format = INPUT_FORMATS[framework]
    for model_status in status.models_status.values():
        model_status_key = model_status.model_config.key
        if model_status_key == input_format:
            continue

        conversion_status = _get_model_status(
            model_status_key=model_status_key,
            models_status=models_status,
            package_path=package_path,
            input_format=input_format,
        )
        if conversion_status == "SKIPPED":
            continue

        export_formats = [fmt.value for fmt in EXPORT_FORMATS[framework]]
        if model_status_key == input_format.value:
            format_files = FORMAT_FILES
        elif any(model_status_key.startswith(fmt) for fmt in export_formats):
            format_files = FORMAT_FILES + ["reproduce_export.sh", "reproduce_export.py"]
        else:
            format_files = FORMAT_FILES + ["reproduce_conversion.sh"]

        if conversion_status != "FAIL":
            format_files += EVALUATION_FILES

        for file in format_files:
            file_path = os.path.join(model_status_key, file)
            files.append(file_path)

    return files


def collect_optimize_status(status: Status) -> Dict:
    test_status = {}
    for model_key, models_status in status.models_status.items():
        for runner_name, runner_status in models_status.runners_status.items():
            key = f"{model_key}.{runner_name}"
            test_status[key] = runner_status.status[Performance.__name__].name

    return test_status


def collect_profile_results(results: ProfilingResults, sample_count: int = 1) -> Dict:
    test_status = {}
    for model_key, runner_results in results.models.items():
        for runner_name, runner_status in runner_results.runners.items():
            key = f"{model_key}.{runner_name}"
            number_of_keys = len(runner_status.detailed.keys())
            expected_value = 0 if runner_status.status != CommandStatus.OK else sample_count

            if number_of_keys != expected_value:
                raise ValueError(
                    f"""Number of keys {number_of_keys} for {key} """
                    f"""is not equal expected value: {expected_value}. """
                    f"""Collected data: {runner_status.detailed}."""
                )

            test_status[key] = runner_status.status.value

    return test_status


def validate_status(status: Dict, expected_statuses: List) -> None:
    current_statuses = set(status.keys())

    if not all(expected_status in current_statuses for expected_status in expected_statuses):
        missing_statuses = set(expected_statuses) - current_statuses
        unexpected_statuses = current_statuses - set(expected_statuses)
        raise ValidationError(
            """Expected statuses not match current statuses.\n """
            f"""Expected: {expected_statuses}\n"""
            f"""Current: {current_statuses}\n"""
            f"""Missing: {missing_statuses}\n"""
            f"""Unexpected: {unexpected_statuses}\n"""
        )


def validate_package(package_path: Union[str, pathlib.Path], expected_files: List[str]) -> None:
    package_path = pathlib.Path(package_path)
    with zipfile.ZipFile(package_path, "r") as zf:
        files = zf.namelist()
        missing = [f for f in expected_files if f not in files]

    if len(missing) > 0:
        missing_str = "\n  ".join(missing)
        raise ValidationError(f"The following files are missing in package:\n {missing_str}")


def validate_model_repository(model_repository: Union[str, pathlib.Path], model_name: str, model_version: str = "1"):
    model_repository = pathlib.Path(model_repository)

    model_path = model_repository / model_name
    assert model_path.exists() is True
    assert model_path.is_dir() is True

    config_path = model_path / "config.pbtxt"
    assert config_path.exists() is True
    assert config_path.is_file() is True

    version_path = model_path / model_version
    assert version_path.exists() is True
    assert version_path.is_dir() is True
    assert any(version_path.iterdir()) is True
