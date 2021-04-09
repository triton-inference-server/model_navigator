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
from typing import Any, List, Optional

import abc
from enum import Enum

import yaml

from ..model_navigator_exceptions import ModelNavigatorException
from ..record.record import RecordType
from .config_field import ConfigField
from .config_list_generic import ConfigListGeneric
from .config_list_numeric import ConfigListNumeric
from .config_list_string import ConfigListString
from .config_object import ConfigObject
from .config_primitive import ConfigPrimitive
from .config_union import ConfigUnion


def _objective_list_output_mapper(objectives):
    # Takes a list of objectives and maps them
    # into a dict
    output_dict = {}
    for objective in objectives:
        value = ConfigPrimitive(type_=int)
        value.set_value(10)
        output_dict[objective] = value
    return output_dict


class ModelNavigatorBaseConfig(metaclass=abc.ABCMeta):
    def __init__(self, *, required_keys: Optional[List[str]] = None):
        """
        Create a new config.
        """

        self._fields = {}
        self._required_keys = required_keys or []
        self._fill_config()

    def __getattr__(self, name):
        try:
            return self._fields[name].value()
        except KeyError:
            raise AttributeError(f"Missing {name} attribute")

    def __setattr__(self, name: str, value: Any) -> None:
        if "_fields" in vars(self) and name in self._fields:
            if isinstance(value, List) and isinstance(value[0], Enum):
                value = [v.value for v in value]
            elif isinstance(value, Enum):
                value = value.value
            self._fields[name].set_value(value)
        else:
            super().__setattr__(name, value)

    def __getstate__(self):
        return self._fields

    def __setstate__(self, state):
        self._fields = state

    def set_config_values(self, args):
        """
        Set the config values. This function sets all the values for the
        config. CLI arguments have the highest priority, then YAML config
        values and then default values.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments from the CLI

        Raises
        ------
        ModelNavigatorException
            If the required fields are not specified, it will raise
            this exception
        """

        # Config file has been specified
        if "config_file" in args:
            yaml_config = self._load_config_file(args.config_file)
        else:
            yaml_config = None
        for key, value in self._fields.items():
            self._fields[key].set_name(key)
            if key in args:
                self._fields[key].set_value(getattr(args, key))
            elif yaml_config is not None and key in yaml_config:
                self._fields[key].set_value(yaml_config[key])
            elif value.default_value() is not None:
                self._fields[key].set_value(value.default_value())
            elif value.required():
                flags = ", ".join(value.flags())
                raise ModelNavigatorException(
                    f"Config for {value.name()} is not specified. You need to specify it using the YAML config file or using the {flags} flags in CLI."
                )

    def get_config(self):
        """
        Get the config dictionary.

        Returns
        -------
        dict
            Returns a dictionary where the keys are the
            configuration name and the values are ConfigField objects.
        """

        return self._fields

    def get_all_config(self):
        """
        Get a dictionary containing all the configurations.

        Returns
        -------
        dict
            A dictionary containing all the configurations.
        """

        config_dict = {}
        for config in self._fields.values():
            config_dict[config.name()] = config.value()

        return config_dict

    def _add_config(self, config_field):
        """
        Add a new config field.

        Parameters
        ----------
        config_field : ConfigField
            Config field to be added

        Raises
        ------
        KeyError
            If the field already exists, it will raise this exception.
        """

        if config_field.name() in self._required_keys:
            config_field.set_required(True)

        if config_field.name() not in self._fields:
            self._fields[config_field.name()] = config_field
        else:
            raise KeyError

    def _load_config_file(self, file_path):
        """
        Load YAML config

        Parameters
        ----------
        file_path : str
            Path to the Model Navigator config file
        """

        with open(file_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            return config

    def merge(self, other_config: "ModelNavigatorBaseConfig"):
        conflicting_fields = {
            name: field
            for name, field in other_config.get_config().items()
            if name in self.get_config() and self.get_config()[name] != other_config.get_config()[name]
        }
        if conflicting_fields:
            raise KeyError(f"Conflicting fields found during configuration merge: {', '.join(conflicting_fields)}")

        self._fields.update(**other_config._fields)

        return self

    @abc.abstractmethod
    def _fill_config(self):
        pass

    def __str__(self) -> str:
        return ", ".join([f"{key}={field.value()}" for key, field in self._fields.items()])


class ModelNavigatorConfig(ModelNavigatorBaseConfig):
    """
    Model Navigator config object.
    """

    def _fill_config(self):
        # common config fields
        self._add_config(
            ConfigField(
                "model_name",
                flags=["--model-name"],
                field_type=ConfigPrimitive(str, required=True),
                description="Name of model.",
            )
        )

        self._add_config(
            ConfigField(
                "model_path",
                flags=["--model-path"],
                default_value="",
                field_type=ConfigPrimitive(str, required=True),
                description="Path to file with model.",
            )
        )
        self._add_config(
            ConfigField(
                "config_file",
                field_type=ConfigPrimitive(str),
                flags=["-f", "--config-file"],
                description="Path to Model Navigator Config File.",
            )
        )
        self._add_config(
            ConfigField(
                "workspace_path",
                flags=["--workspace-path"],
                default_value="workspace",
                field_type=ConfigPrimitive(str),
                description="Path to output directory.",
            )
        )
        self._add_config(
            ConfigField(
                "verbose",
                field_type=ConfigPrimitive(bool),
                parser_args={"action": "store_true"},
                default_value=False,
                flags=["--verbose"],
                description="Enable verbose mode.",
            )
        )

        # analyzer specific fields
        self._add_config(
            ConfigField(
                "top_n_configs",
                field_type=ConfigPrimitive(int),
                flags=["--top-n-configs"],
                description="Number of top final configurations selected from analysis.",
                default_value=3,
            )
        )

        self._add_config(
            ConfigField(
                "max_concurrency",
                flags=["--max-concurrency"],
                field_type=ConfigPrimitive(int),
                default_value=1024,
                description="Max concurrency used for config search in analysis",
            )
        )
        self._add_config(
            ConfigField(
                "max_instance_count",
                flags=["--max-instance-count"],
                field_type=ConfigPrimitive(int),
                default_value=5,
                description="Max number of model instances used for config search in analysis",
            )
        )
        self._add_config(
            ConfigField(
                "max_preferred_batch_size",
                flags=["--max-preferred-batch-size"],
                field_type=ConfigPrimitive(int),
                default_value=32,
                description="Max preferred batch size used for config search in analysis",
            )
        )

        # User defined list of parameters
        self._add_config(
            ConfigField(
                "concurrency",
                flags=["--concurrency"],
                field_type=ConfigListNumeric(int),
                description="Concurrency values to be used for the analysis in manual mode",
            )
        )

        self._add_config(
            ConfigField(
                "instance_counts",
                flags=["--instance-counts"],
                field_type=ConfigListNumeric(int),
                description="Comma-delimited list of instance counts to use for the profiling.",
            )
        )
        self._add_config(
            ConfigField(
                "preferred_batch_sizes",
                flags=["--preferred-batch-sizes"],
                field_type=ConfigListGeneric(ConfigListNumeric(int)),
                description="List of batch sizes to use for the profiling in dynamic batching.",
            )
        )

        self._add_config(
            ConfigField(
                "max_latency_ms",
                flags=["--max-latency-ms"],
                field_type=ConfigPrimitive(int),
                description="Maximal latency in ms that analyzed model should match.",
            )
        )

        self._add_config(
            ConfigField(
                "min_throughput",
                flags=["--min-throughput"],
                field_type=ConfigPrimitive(int),
                description="Minimal throughput that analyzed model should match.",
            )
        )
        self._add_config(
            ConfigField(
                "max_gpu_usage_mb",
                flags=["--max-gpu-usage-mb"],
                field_type=ConfigPrimitive(int),
                description="Maximal GPU memory usage in MB that analyzed model should match.",
            )
        )

        objectives_scheme = ConfigUnion(
            [
                ConfigObject(
                    schema={tag: ConfigPrimitive(type_=int) for tag in RecordType.get_all_record_types().keys()}
                ),
                ConfigListString(output_mapper=_objective_list_output_mapper),
            ]
        )

        self._add_config(
            ConfigField(
                "objectives",
                field_type=objectives_scheme,
                default_value={"perf_throughput": 10},
                description="Model Navigator uses the objectives described here to find the best configuration for model.",
            )
        )

        self._add_config(
            ConfigField(
                "triton_version",
                flags=["--triton-version"],
                field_type=ConfigPrimitive(str),
                default_value="21.03-py3",
                description="Triton Server Docker version",
            )
        )
        self._add_config(
            ConfigField(
                "triton_launch_mode",
                field_type=ConfigPrimitive(str),
                flags=["--triton-launch-mode"],
                default_value="local",
                choices=["local", "docker"],
                description="The method by which to launch Triton Server. "
                "'local' assumes tritonserver binary is available locally. "
                "'docker' pulls and launches a triton docker container with "
                "the specified version.",
            )
        )
        self._add_config(
            ConfigField(
                "triton_server_path",
                field_type=ConfigPrimitive(str),
                flags=["--triton-server-path"],
                default_value="tritonserver",
                description="The full path to the tritonserver binary executable",
            )
        )

        self._add_config(
            ConfigField(
                "client_protocol",
                flags=["--client-protocol"],
                choices=["http", "grpc"],
                field_type=ConfigPrimitive(str),
                default_value="grpc",
                description="The protocol used to communicate with the Triton Inference Server",
            )
        )
        self._add_config(
            ConfigField(
                "gpus",
                flags=["--gpus"],
                field_type=ConfigListString(),
                default_value="all",
                description="List of GPU UUIDs to be used for the profiling. "
                "Use 'all' to profile all the GPUs visible by CUDA.",
            )
        )
