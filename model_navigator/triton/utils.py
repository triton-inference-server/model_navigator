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
import logging
from typing import Tuple

from model_navigator.exceptions import ModelNavigatorDeployerException
from model_navigator.model import ModelSignatureConfig

LOGGER = logging.getLogger(__name__)


def parse_server_url(server_url: str) -> Tuple[str, str, int]:
    DEFAULT_PORTS = {"http": 8000, "grpc": 8001}

    # extract protocol
    server_url_items = server_url.split("://")
    if len(server_url_items) != 2:
        raise ValueError("Prefix server_url with protocol ex.: grpc://127.0.0.1:8001")
    requested_protocol, server_url = server_url_items
    requested_protocol = requested_protocol.lower()

    if requested_protocol not in DEFAULT_PORTS:
        raise ValueError(f"Unsupported protocol: {requested_protocol}")

    # extract host and port
    default_port = DEFAULT_PORTS[requested_protocol]
    server_url_items = server_url.split(":")
    if len(server_url_items) == 1:
        host, port = server_url, default_port
    elif len(server_url_items) == 2:
        host, port = server_url_items
        port = int(port)
        if port != default_port:
            LOGGER.warning(
                f"Current server URL is {server_url} while default {requested_protocol} port is {default_port}"
            )
    else:
        raise ValueError(f"Could not parse {server_url}. Example of correct server URL: grpc://127.0.0.1:8001")
    return requested_protocol, host, port


def rewrite_signature_to_model_config(model_config, signature: ModelSignatureConfig):
    from model_navigator.triton.client import client_utils, grpc_client

    if not signature.inputs or not signature.outputs:
        raise ModelNavigatorDeployerException(
            "Signature is required to create Triton Model Configuration. Could not obtain it."
        )

    def _rewrite_io_spec(spec_, item):
        dtype = f"TYPE_{client_utils.np_to_triton_dtype(spec_.dtype)}"
        dims = [1] if len(spec_.shape) <= 1 else spec_.shape[1:]  # do not pass batch size

        item.name = spec_.name
        item.dims.extend(dims)
        item.data_type = getattr(grpc_client.model_config_pb2, dtype)
        if len(spec_.shape) <= 1:
            item.reshape.shape.extend([])

    for _name, spec in signature.inputs.items():
        input_item = model_config.input.add()
        _rewrite_io_spec(spec, input_item)

    for _name, spec in signature.outputs.items():
        output_item = model_config.output.add()
        _rewrite_io_spec(spec, output_item)


def get_shape_params(dataset_profile_config):
    if not dataset_profile_config.max_shapes:
        return None

    def _shape_param_format(name, shape_):
        return f"{name}:{','.join(map(str, shape_[1:]))}"

    shapes_param = [_shape_param_format(name, shape_) for name, shape_ in dataset_profile_config.max_shapes.items()]

    return shapes_param
