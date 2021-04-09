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
from copy import deepcopy

from ..constants import CONFIG_PARSER_FAILURE, CONFIG_PARSER_SUCCESS
from .config_status import ConfigStatus
from .config_value import ConfigValue


def _default_validator(x, value):
    if type(x) is list and len(x) > 0:
        return ConfigStatus(CONFIG_PARSER_SUCCESS)

    return ConfigStatus(
        CONFIG_PARSER_FAILURE,
        f'The value for field "{value.name()}" should be a list' " and the length must be larger than zero.",
    )


class ConfigListGeneric(ConfigValue):
    """
    A generic list.
    """

    def __init__(self, type_, preprocess=None, required=False, validator=None, output_mapper=None, name=None):
        """
        Create a new list of generic objects.

        Parameters
        ----------
        type_ : ConfigValue
            The type of elements in the list
        preprocess : callable
            Function be called before setting new values.
        required : bool
            Whether a given config is required or not.
        validator : callable or None
            A validator for the final value of the field.
        output_mapper: callable
            This callable unifies the output value of this field.
        name : str
            Fully qualified name for this field.
        """

        validator = validator or functools.partial(_default_validator, value=self)
        super().__init__(preprocess, required, validator, output_mapper, name)

        # type_ should be instance of ConfigValue
        assert isinstance(type_, ConfigValue)

        self._type = type_
        self._cli_type = str
        self._value = []
        self._output_mapper = output_mapper

    def set_value(self, value):
        """
        Set the value for this field.

        Parameters
        ----------
        value : object
            The value for this field.
        """
        type_ = self._type

        new_value = []
        if type(value) is list:
            for item in value:
                list_item = deepcopy(type_)
                config_status = list_item.set_value(item)
                if config_status.status() == CONFIG_PARSER_SUCCESS:
                    new_value.append(list_item)
                else:
                    return config_status
        else:
            return ConfigStatus(
                status=CONFIG_PARSER_FAILURE,
                message=f'Value for field "{self.name()}" must be a list, value is "{value}".',
                config_object=self,
            )

        return super().set_value(new_value)

    def set_name(self, name):
        super().set_name(name)
        self._type.set_name(name)
