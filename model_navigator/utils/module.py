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
"""Module related utilities."""

import importlib
from types import ModuleType
from typing import Any, Optional

from model_navigator.core.logger import LOGGER

MODULE_VAR_NAME = "module"


def lazy_import(name: str):
    """Lazy load module with given name.

    Args:
        name: Name of module to

    Returns:
        Imported lazy module
    """

    def _import_module() -> Optional[ModuleType]:
        module = None
        try:
            module = importlib.import_module(name)
        except ImportError as err:
            LOGGER.error(f"Module: {name!r} is required but could not be imported.\nError: {err}\n")

        return module

    class LazyModule:
        """Lazy module using custom import."""

        def __init__(self):
            """Initialize lazy module."""
            super().__setattr__(MODULE_VAR_NAME, None)

        def __import_module(self):
            """Magic method for initializing the module object."""
            if self.module is None:
                super().__setattr__(MODULE_VAR_NAME, _import_module())
            return self.module

        def __getattr__(self, name: str) -> Any:
            """Get lazy module attribute.

            Args:
                name: attribute name

            Returns:
                Value of attribute
            """
            module = self.__import_module()
            return getattr(module, name)

        def __setattr__(self, name: str, value: Any):
            """Set lazy module attribute.

            Args:
                name: attribute name
                value: attribute value
            """
            module = self.__import_module()
            return setattr(module, name, value)

    return LazyModule()
