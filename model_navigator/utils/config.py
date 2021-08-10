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
import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

import dacite
import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "model_navigator.yaml"

_MISSING = "__MISSING__"


def monkeypatch_dataclasses():
    # monkey patch dataclasses namedtuple conversion,
    # till wait for fix in https://github.com/ericvsmith/dataclasses/issues/151
    orig_asdict_inner = dataclasses._asdict_inner  # pytype: disable=module-attr

    def _asdict_inner(obj, dict_factory):
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*(_asdict_inner(v, dict_factory) for v in obj))
        else:
            return orig_asdict_inner(obj, dict_factory)

    dataclasses._asdict_inner = _asdict_inner

    orig_astuple_inner = dataclasses._astuple_inner  # pytype: disable=module-attr

    def _astuple_inner(obj, tuple_factory):
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*(_astuple_inner(v, tuple_factory) for v in obj))
        else:
            return orig_astuple_inner(obj, tuple_factory)

    dataclasses._astuple_inner = _astuple_inner


monkeypatch_dataclasses()


def dataclass2dict(config):
    def _dict_factory_with_enum_values_extraction(fields_):
        result = []
        for key_, value_ in fields_:
            if isinstance(key_, Enum):
                key_ = key_.value

            if isinstance(value_, Enum):
                value_ = value_.value
            elif isinstance(value_, Path):
                value_ = value_.as_posix()
            elif isinstance(value_, np.dtype):
                value_ = str(value_)
            elif isinstance(value_, (tuple, list)):
                value_ = [v.value if isinstance(v, Enum) else v for v in value_]
            elif isinstance(value_, dict):
                value_ = _dict_factory_with_enum_values_extraction(value_.items())
            result.append((key_, value_))
        return dict(result)

    init_fields_names = [field.name for field in dataclasses.fields(config) if field.init]

    new_config_dict = dataclasses.asdict(config, dict_factory=_dict_factory_with_enum_values_extraction)
    config_dict_with_only_init_items = {k: v for k, v in new_config_dict.items() if k in init_fields_names}
    return config_dict_with_only_init_items


def dict2dataclass(cls, data):
    fields_names = [f.name for f in dataclasses.fields(cls)]
    probable_data = {k: v for k, v in data.items() if k in fields_names}
    LOGGER.debug(f"Parsing {probable_data} {cls}")
    return dacite.from_dict(cls, data, config=dacite.Config(cast=[Enum, Path, tuple, np.dtype]))


@dataclasses.dataclass
class BaseConfig:
    def __post_init__(self):
        from model_navigator.utils.cli import is_namedtuple

        fields = dataclasses.fields(self)
        if any([is_namedtuple(field.type) for field in fields]):
            raise TypeError("Do not use NamedTuples as fields")

    @classmethod
    def from_dict(cls, data):
        try:
            return dict2dataclass(cls, data)
        except dacite.MissingValueError as e:
            raise TypeError(e)
        except dacite.WrongTypeError as e:
            raise ValueError(e)


class ConfigFile(ABC):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save_config(self, config: BaseConfig):
        pass

    @abstractmethod
    def load(self, cls) -> BaseConfig:
        pass


class PyYamlAdapter:
    def load(self, stream):
        return yaml.safe_load(stream)

    def dump(self, obj, stream):
        yaml.dump(obj, stream, sort_keys=False)


class RuamelYamlAdapter:
    def __init__(self):
        # use ruamel yaml because empty key in mapping is serialized strangely in pyyaml (see atol/rtol)

        try:
            from ruamel.yaml import YAML  # pytype: disable=import-error
        except ImportError:
            from ruamel_yaml import YAML  # pytype: disable=import-error

        self._yaml = YAML()

    def load(self, stream):
        return self._yaml.load(stream)

    def dump(self, obj, stream):
        self._yaml.dump(obj, stream)


class YamlConfigFile(ConfigFile):
    def __init__(self, config_path: Union[str, Path]) -> None:
        self._config_path = Path(config_path)
        self._yaml_adapter = RuamelYamlAdapter()
        self._yaml_adapter = PyYamlAdapter()
        self._config_dict = self._load(self._config_path)

    def _load(self, config_path: Path):
        config_dict = {}
        if config_path.exists():
            with config_path.open("r") as config_file:
                config_dict = self._yaml_adapter.load(config_file)
        return config_dict

    def _flush(self):
        if self._config_dict:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w") as config_file:
                self._yaml_adapter.dump(self._config_dict, config_file)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def save_config(self, config: BaseConfig):
        new_config_dict = dataclass2dict(config)
        for name, value in new_config_dict.items():
            old_value = self._config_dict.get(name, _MISSING)
            if old_value != _MISSING and value != old_value:
                raise ValueError(
                    f"There is already {name}={value} conflicts with {name}={old_value} "
                    f"already present in {self._config_path} config file"
                )
            self._config_dict[name] = value

        self._flush()

    def save_key(self, name: str, value):
        old_value = self._config_dict.get(name, _MISSING)
        if old_value != _MISSING and value != old_value:
            raise ValueError(
                f"There is already {name}={value} conflicts with {name}={old_value} "
                f"already present in {self._config_path} config file"
            )
        self._config_dict[name] = value
        self._flush()

    def load(self, cls):
        return cls.from_dict(self._config_dict)

    @property
    def config_dict(self):
        return copy.deepcopy(self._config_dict)

    def close(self):
        self._flush()
