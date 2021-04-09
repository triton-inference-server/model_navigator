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
from ..model_navigator_exceptions import ModelNavigatorException


class ModelAnalyzerConfig:
    """
    A config class to set arguments to the Model Analyzer.
    An argument set to None will use the default.
    """

    def __init__(self):
        # Args will be a dict with the string representation as key
        self._args = {"log-level": "INFO", "override-output-model-repository": True}

        self._options = {
            "-f": "config.yaml",
        }

        self._input_to_options = {
            "config-file": "-f",
        }

    def to_cli_string(self):
        """
        Utility function to convert a config into a
        string of arguments to the server with CLI.
        Returns
        -------
        str
            the command consisting of all set arguments to
            the model analyzer.
            e.g. '--model-repository=/models --log-verbose=True'
        """
        # single dashed options, then verbose flags, then main args
        args = [f"{k} {v}" for k, v in self._options.items() if v]

        for key, val in self._args.items():
            if not val:
                continue

            if isinstance(val, bool):
                args.append(f"--{key}")
            else:
                args.append(f"--{key}={val}")

        return " ".join(args)

    def __getitem__(self, key):
        """
        Gets an arguments value in config
        Parameters
        ----------
        key : str
            The name of the argument to the model analyzer
        Returns
        -------
            The value that the argument is set to in this config
        """

        if key in self._args:
            return self._args[key]
        elif key in self._input_to_options:
            return self._options[self._input_to_options[key]]
        else:
            raise ModelNavigatorException(f"'{key}' Key not found in config")

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.
        Parameters
        ----------
        key : str
            The name of the argument to the model analyzer
        value : (any)
            The value to which the argument is being set
        Raises
        ------
        TritonModelAnalyzerException
            If key is unsupported or undefined in the
            config class
        """
        if key in self._args:
            self._args[key] = value
        elif key in self._input_to_options:
            self._options[self._input_to_options[key]] = value
        else:
            raise ModelNavigatorException(
                f"The argument '{key}' to the Model Analyzer is not supported by the Model Navigator."
            )
