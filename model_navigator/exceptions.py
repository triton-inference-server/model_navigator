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
from pathlib import Path
from typing import Optional

from click import ClickException


class ModelNavigatorException(Exception):
    def __init__(self, message: str, log_path: Optional[Path] = None):
        self._message = message
        self._log_path = log_path

    def __str__(self):
        return self._message

    @property
    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        return self._message

    @property
    def log_path(self):
        return self._log_path


class ModelNavigatorDeployerException(ModelNavigatorException):
    pass


class BadParameterModelNavigatorDeployerException(ModelNavigatorDeployerException):
    pass


class ModelNavigatorConverterException(ModelNavigatorException):
    pass


class ModelNavigatorConverterCommandException(ModelNavigatorException):
    pass


class ModelNavigatorProfileException(ModelNavigatorException):
    pass


class ModelNavigatorAnalyzeException(ModelNavigatorException):
    pass


class ModelNavigatorCliException(ClickException):
    pass


class ModelNavigatorInvalidPackageException(ClickException):
    pass
