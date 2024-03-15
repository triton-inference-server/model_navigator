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
"""Model Navigator exceptions module."""

import pathlib
from typing import Optional


class ModelNavigatorError(Exception):
    """Base exception for Model Navigator exceptions."""

    def __init__(self, message: str, log_path: Optional[pathlib.Path] = None):
        """Initialize exception object.

        Args:
            message: An error message
            log_path: A path to log file to store logs
        """
        self._message = message
        self._log_path = log_path

    def __str__(self):
        """Convert exception object to string.

        Returns:
            Error message of exception
        """
        return self._message

    @property
    def message(self) -> str:
        """Get the exception message.

        Returns:
            The message associated with this exception, or None if no message.
        """
        return self._message

    @property
    def log_path(self) -> pathlib.Path:
        """Get the log file path.

        Returns:
            The path to file where logs are stored, or None if no path.
        """
        return self._log_path


class ModelNavigatorWarning(ModelNavigatorError, Warning):
    """ModelNavigatorWarning exception."""

    pass


class ModelNavigatorRuntimeError(ModelNavigatorError):
    """ModelNavigatorRuntimeError exception."""

    pass


class ModelNavigatorUserInputError(ModelNavigatorError):
    """ModelNavigatorUserInputError exceptions.

    Raised when provided input data by user if not valid.
    """

    pass


class ModelNavigatorNotFoundError(ModelNavigatorError):
    """ModelNavigatorNotFoundError exceptions.

    Raised when for provided configuration model or runner was not found.
    """

    pass


class ModelNavigatorBackwardCompatibilityError(ModelNavigatorError):
    """ModelNavigatorBackwardCompatibilityError exceptions.

    Raised when backward compatibility is broken and user must upgrade Python or Navigator package.
    """

    pass


class ModelNavigatorDeployerError(ModelNavigatorError):
    """ModelNavigatorDeployerError exception."""

    pass


class ModelNavigatorWrongParameterError(ModelNavigatorError):
    """ModelNavigatorWrongParameterError exception."""

    pass


class ModelNavigatorRuntimeAnalyzerError(ModelNavigatorError):
    """ModelNavigatorRuntimeAnalyzerException exception."""

    pass


class ModelNavigatorMissingSourceModelError(ModelNavigatorError):
    """ModelNavigatorMissingSourceModelException exception."""

    pass


class ModelNavigatorEmptyPackageError(ModelNavigatorError):
    """ModelNavigatorEmptyPackageException exception."""

    pass


class ModelNavigatorTensorTypeError(ModelNavigatorError):
    """ModelNavigatorTensorTypeError exception."""

    pass


class ModelNavigatorProfilingError(ModelNavigatorError):
    """ModelNavigatorProfilingError exception."""

    pass


class ModelNavigatorConfigurationError(ModelNavigatorError):
    """Raised when the configuration is not valid."""

    pass


class ModelNavigatorConfigurationWarning(ModelNavigatorWarning):
    """Raised when the configuration is not valid, but the optimize command can still be run without ambiguity."""

    pass


class ModelNavigatorCommandNotExecutable(ModelNavigatorError):
    """Raised when command execution is not possible."""

    pass


class ModelNavigatorModuleNotOptimizedError(ModelNavigatorError):
    """Raised when the module is not optimized and is required to be optimized."""

    pass
