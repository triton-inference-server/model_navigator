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
import functools

from model_navigator.constants import CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS

from .config_status import ConfigStatus
from .config_value import ConfigValue


def _default_validator(x, value):
    if type(x) is list and len(x) > 0:
        return ConfigStatus(CONFIG_PARSER_SUCCESS)

    return ConfigStatus(
        CONFIG_PARSER_FAILURE,
        f'The value for field "{value.name()}" should be a list' " and the length must be larger than zero.",
    )


class ConfigListString(ConfigValue):
    """
    A list of string values.
    """

    def __init__(self, preprocess=None, required=False, validator=None, output_mapper=None, name=None):
        """
        Instantiate a new ConfigListString

        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the value of the field.
        output_mapper: callable or None
            This callable unifies the output value of this field.
        name : str
            Fully qualified name for this field.
        """

        validator = validator or functools.partial(_default_validator, value=self)
        super().__init__(preprocess, required, validator, output_mapper, name)
        self._type = self._cli_type = str
        self._value = []

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field. It can be a string of comma-delimited
            items or a list.

        Returns
        -------
        int
            1 on success, and 0 on failure
        """
        new_value = []
        if self._is_string(value):
            value = value.split(",")
            for item in value:
                new_value.append(self._type(item))
        elif self._is_list(value):
            for item in value:
                if not self._is_primitive(item):
                    return ConfigStatus(
                        CONFIG_PARSER_FAILURE,
                        "The value for each item in the list should"
                        f' be a primitive value not "{item}" for field '
                        f'"{self.name()}".',
                        self,
                    )
                new_value.append(self._type(item))
        else:
            if self._is_dict(value):
                return ConfigStatus(
                    CONFIG_PARSER_FAILURE,
                    f'The value for field "{self.name()}" should not be'
                    " a dictionary, current "
                    f'value is "{value}".',
                    self,
                )
            new_value = [self._type(value)]

        return super().set_value(new_value)
