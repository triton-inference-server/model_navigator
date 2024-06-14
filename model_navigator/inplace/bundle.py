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
"""Bundle is a cache archive - collection of files from model navigator cache stored as a single file.

Enables easy storage and sharing of optimized models and modules.

Selecting modules strategies - `nav.bundle.save` `modules` argument:
 * `nav.bundle.BestRunnersSelection(runner_selection_strategy=...)` - default, selects only best runners from registered modules for bundling.
 * `nav.bundle.RegisteredModulesSelection()` - selects only registered modules for bundling
 * `nav.bundle.ModulesByNameSelection(module_names=[...])` - selects only selected registered modules with given names for bundling
 * `nav.bundle.AllModulesSelection()` - selects all modules from cache for bundling

Example:
    Saving cache bundle:
    ```python
    import model_navigator as nav

    # all modules
    nav.bundle.save("./bundle.nav", modules=nav.bundle.AllModulesSelection())

    # saving just registered modules - nav.Module()
    nav.bundle.save("./bundle.nav")
    # equivalent to ...
    nav.bundle.save("./bundle.nav", modules=nav.bundle.BestRunnersSelection())
    #  ... and ...
    nav.bundle.save("./bundle.nav", modules=nav.bundle.BestRunnersSelection(runner_selection_strategy=nav.configuration.MaxThroughputAndMinLatencyStrategy())

    # saving all registered modules (with all runners)
    nav.bundle.save("./bundle.nav", modules=nav.bundle.RegisteredModulesSelection())

    # saving just selected registered modules
    nav.bundle.save("./bundle.nav", modules=nav.bundle.ModulesByNameSelection(["module1", "module2"]))

    # saving with tags
    nav.bundle.save("./bundle.nav", tags=["batch=large"])
    ```

    Loading cache bundle:
    ```python
    import model_navigator as nav

    # loading bundle
    nav.bundle.load("./bundle.nav")

    # loading bundle with tags
    nav.bundle.load("./bundle.nav", tags=["batch=large"])

    # loading bundle with force - skips environment check
    nav.bundle.load("./bundle.nav", force=True)
    ```

"""

import abc
import os
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Union

import yaml
from packaging.version import Version

from model_navigator.configuration import MaxThroughputAndMinLatencyStrategy, RuntimeSearchStrategy
from model_navigator.core.logger import LOGGER
from model_navigator.exceptions import ModelNavigatorConfigurationError, ModelNavigatorModuleNotOptimizedError
from model_navigator.inplace.config import inplace_cache_dir
from model_navigator.inplace.registry import module_registry
from model_navigator.utils.common import get_default_status_filename
from model_navigator.utils.environment import get_env


def load(bundle_path: Union[str, Path], tags: Optional[List[str]] = None, force: bool = False):
    """Loads bundle from archive file. Extracts files to cache directory.

    Args:
        bundle_path (Union[str, Path]): A path to archive.
        tags (Optional[list[str]], optional):  user specified bundle info, if not None tags MUST match. Defaults to None.
        force (bool, optional): If True, omits environment matching step. Defaults to False.

    Raises:
        ModelNavigatorConfigurationError: If bundle does not match current environment
    """
    cache_dir = inplace_cache_dir()

    # check if bundle is matching current environment and tags
    if not (force or is_matching(bundle_path, tags)):
        raise ModelNavigatorConfigurationError(
            f"Bundle does not match current environment('{bundle_path}'). Run DEBUG mode and see logs."
        )

    with zipfile.ZipFile(bundle_path, "r") as zip_file:
        # remove all modules from cache that are in the bundle
        module_names = {Path(name).parts[0] for name in zip_file.namelist()}
        for name in module_names:
            if name != ".":
                shutil.rmtree(cache_dir / name, ignore_errors=True)

        # extract bundle to cache
        zip_file.extractall(cache_dir)


def is_matching(bundle_path: Union[str, Path], tags: Optional[List[str]] = None) -> bool:
    """Validates bundle archive file. Extracts modules status metadata and compares it with current HW and SW stack.

    Use case: If user want to use bundle on the different machine, for TRT model plan,
        * GPU must match
        * CUDA version must match
        * TensorRT version must match

    Args:
        bundle_path (Optional[str, Path]): Path to archive
        tags (Optional[List[str]]): user specified bundle info, if not None tags MUST match

    Returns:
        bool: if archive attributes match HW architecture and SW stack
    """
    current_env = get_env()

    os_names = current_env["os"]["platform"]
    gpu_names = current_env["gpu"]["name"]
    cuda_versions = _major_minor_version(current_env["gpu"]["cuda_version"])
    trt_versions = _major_minor_version(current_env["python_packages"].get("tensorrt", "0.0"))

    with zipfile.ZipFile(bundle_path, "r") as zip_file:
        # check if tags match
        if tags:
            with zip_file.open("tags.yaml") as tags_file:
                bundle_tags = yaml.safe_load(tags_file)
                if set(bundle_tags["tags"]) != set(tags):
                    LOGGER.warn(f"TAGS mismatch in bundle '{bundle_path}'")
                    return False

        # expecting status.yaml files in the modules dirs
        for status_file in (name for name in zip_file.namelist() if name.endswith("status.yaml")):
            with zip_file.open(status_file) as status:
                status = yaml.safe_load(status)

                # basic check is if OS matches
                if status["environment"]["os"]["platform"] != os_names:
                    LOGGER.warn(f"OS mismatch in {status_file}")
                    return False

                # TensorRT runners require plan optimized for specific GPU, CUDA, and TensorRT versions
                if _has_module_trt_runner(status["module_status"]):
                    if status["environment"]["gpu"]["name"] != gpu_names:
                        LOGGER.warn(f"GPU mismatch in {status_file}")
                        return False
                    if _major_minor_version(status["environment"]["gpu"]["cuda_version"]) != cuda_versions:
                        LOGGER.warn(f"CUDA version mismatch in {status_file}")
                        return False
                    if _major_minor_version(status["environment"]["python_packages"]["tensorrt"]) != trt_versions:
                        LOGGER.warn(f"TRT version mismatch in {status_file}")
                        return False
                else:
                    LOGGER.debug(f"No TRT runner in {status_file}")

    return True


def _major_minor_version(version: str) -> Tuple[int, int]:
    try:
        parsed_version = Version(version)
        return parsed_version.major, parsed_version.minor
    except ValueError:
        LOGGER.error(f"Bundle check: Cannot parse version '{version}'")
        return -1, 0


def _has_module_trt_runner(module_status) -> bool:
    for sub_modules in module_status.values():
        for runners in sub_modules["models_status"].values():
            if runners["model_config"]["format"] == "trt":
                return True
    return False


class BundleModuleSelection(abc.ABC):  # noqa: B024
    """Module and runner selection strategy for bundle creation."""

    def __str__(self) -> str:
        """Name of the selection strategy."""
        return self.__class__.__name__


class AllModulesSelection(BundleModuleSelection):
    """Selects all modules from cache for bundling."""


class RegisteredModulesSelection(BundleModuleSelection):
    """Selects only registered modules from cache for bundling."""


class BestRunnersSelection(BundleModuleSelection):
    """Selects only best runners from registered modules for bundling."""

    def __init__(self, runner_selection_strategy: Optional[RuntimeSearchStrategy] = None) -> None:
        """Init.

        Args:
            runner_selection_strategy (Optional[RuntimeSearchStrategy], optional): Module loading strategy to use during modules selection for save. Defaults to MaxThroughputAndMinLatencyStrategy.
        """
        super().__init__()
        self.runner_selection_strategy = runner_selection_strategy or MaxThroughputAndMinLatencyStrategy()


class ModulesByNameSelection(BundleModuleSelection):
    """Sometimes user may want to save only specific modules."""

    def __init__(self, module_names: List[str]) -> None:
        """Init.

        Args:
            module_names (List[str]): List of names of the registered modules to save.
        """
        self.module_names = module_names


def save(
    bundle_path: Union[str, Path],
    modules: Optional[BundleModuleSelection] = None,
    tags: Optional[List[str]] = None,
):
    """Saves cache bundle to archive for easy storage.

    Args:
        bundle_path (Union[str, Path]): Where to save bundle file
        modules (BundleModuleSelection): Strategy for selecting modules. @see BundleModuleSelection and subclasses  Defaults to BestRunnersSelection with MaxThroughputAndMinLatencyStrategy runners.
        tags (Optional[List[str]], optional): a set of tags, for better bundle identification and selection. Defaults to None.

    Raises:
        ModelNavigatorModuleNotOptimizedError: When selected modules are not optimized yet
    """
    modules = modules or BestRunnersSelection(MaxThroughputAndMinLatencyStrategy())
    cache_dir = inplace_cache_dir()

    # saving to temporary file and then moving to final location to avoid corrupted files
    with TemporaryDirectory() as tmp_dir:
        tmp_zip = Path(tmp_dir) / "bundle.nav"
        with zipfile.ZipFile(tmp_zip, "w") as zip_file:
            for entry in _selected_cache_entries(modules):
                entry_path = cache_dir / entry

                if entry_path.is_file():
                    zip_file.write(entry_path, entry)
                else:
                    for dirpath, _, filenames in os.walk(entry_path):  # Path.walk() since 3.12
                        for filename in filenames:
                            file_path = Path(dirpath) / filename
                            zip_file.write(file_path, file_path.relative_to(cache_dir))

            # lastly adding tags to the bundle
            zip_file.writestr("tags.yaml", yaml.dump({"tags": tags or []}))

        shutil.copy(tmp_zip, bundle_path)


def _only_module_best_runner(strategy: RuntimeSearchStrategy) -> List[str]:
    cache_dir = inplace_cache_dir()

    best_modules = []
    for name, module in module_registry.modules.items():
        _raise_if_module_not_optimized(name, module)

        optimized_module = module._wrapper
        for package in optimized_module._packages:
            workspace_path = package.workspace.path

            # TODO is there other way to get files need for runner?
            status_path = workspace_path / get_default_status_filename()
            context_path = workspace_path / "context.yaml"
            nav_log_path = workspace_path / "navigator.log"
            model_input_path = workspace_path / "model_input"
            model_output_path = workspace_path / "model_output"

            runtime_result = package._get_best_runtime(strategy=strategy)
            module_runner_path = workspace_path / runtime_result.model_status.model_config.path.parent

            best_modules += [
                str(status_path.relative_to(cache_dir)),
                str(context_path.relative_to(cache_dir)),
                str(nav_log_path.relative_to(cache_dir)),
                str(model_input_path.relative_to(cache_dir)),
                str(model_output_path.relative_to(cache_dir)),
                str(module_runner_path.relative_to(cache_dir)),
            ]

    return best_modules


def _registered_modules_names() -> List[str]:
    return [name for name, module in module_registry.modules.items() if _raise_if_module_not_optimized(name, module)]


def _all_modules_names() -> List[str]:
    cache_dir = inplace_cache_dir()
    return [module.name for module in cache_dir.iterdir() if module.is_dir()]


def _modules_by_name(module_names: List[str]) -> List[str]:
    # TODO(kn): Can we have also best runners for selected modules by name?
    return [name for name in _registered_modules_names() if name in module_names]


def _selected_cache_entries(select_modules: BundleModuleSelection) -> List[str]:
    if isinstance(select_modules, AllModulesSelection):
        return _all_modules_names()
    if isinstance(select_modules, RegisteredModulesSelection):
        return _registered_modules_names()
    if isinstance(select_modules, BestRunnersSelection):
        return _only_module_best_runner(select_modules.runner_selection_strategy)
    if isinstance(select_modules, ModulesByNameSelection):
        return _modules_by_name(select_modules.module_names)

    raise ValueError(f"Unknown module selection strategy: {select_modules}")


def _raise_if_module_not_optimized(name, module):
    if not module.is_optimized:
        raise ModelNavigatorModuleNotOptimizedError(
            f"Module '{name}' is not optimized. Please optimize all modules before saving."
        )
    return True
