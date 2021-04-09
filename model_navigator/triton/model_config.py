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
from typing import List, NamedTuple, Optional, Union

import logging
from pathlib import Path

import onnx
from google.protobuf.text_format import MessageToString
from polygraphy.backend.onnx import OnnxFromPath
from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata
from model_navigator import Accelerator, Format, Precision
from model_navigator.tensor import IOSpec, TensorSpec

from .client import client_utils, grpc_client

LOGGER = logging.getLogger(__name__)

_PLATFORM_PER_FORMAT = {
    Format.TF_GRAPHDEF: "tensorflow_graphdef",
    Format.TF_SAVEDMODEL: "tensorflow_savedmodel",
    Format.TF_TRT: "tensorflow_savedmodel",
    Format.ONNX: "onnxruntime_onnx",
    Format.TRT: "tensorrt_plan",
    Format.TS_SCRIPT: "pytorch_libtorch",
    Format.TS_TRACE: "pytorch_libtorch",
}


def _load_io_specs(*, exported_model: Union[str, Path]):
    model_path = Path(exported_model)
    yaml_path = model_path.parent / f"{model_path.stem}.yaml"

    try:
        return IOSpec.from_file(yaml_path)
    except OSError:
        raise ValueError(f"TorchScript model configurator expects file {yaml_path} with tensor information.")


class ModelConfig(NamedTuple):
    model_name: str
    model_version: str
    model_format: Format
    max_batch_size: int
    precision: Precision
    gpu_engine_count: int
    preferred_batch_sizes: List[int]
    max_queue_delay_us: int
    capture_cuda_graph: int
    accelerator: Accelerator
    inputs: Optional[List[TensorSpec]]
    outputs: Optional[List[TensorSpec]]

    def validate(self):
        assert self.capture_cuda_graph in [0, 1]
        if Accelerator.AMP == self.accelerator and self.model_format not in [
            Format.TF_GRAPHDEF,
            Format.TF_SAVEDMODEL,
            Format.TF_TRT,
        ]:
            raise ValueError("AMP accelerator is available only for TF formats")
        if Accelerator.TRT == self.accelerator and self.model_format not in [
            Format.ONNX,
            Format.TF_GRAPHDEF,
            Format.TF_SAVEDMODEL,
        ]:
            raise ValueError("TensorRT accelerator is available only for ONNX and TF formats")
        if self.model_format.value.startswith("ts-") and (self.inputs is None or self.outputs is None):
            raise ValueError("TorchScripts models require --io-props parameter")

    def generate_prototxt_payload(self):
        self.validate()

        # https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/protobuf_api/model_config.proto.html
        model_config = grpc_client.model_config_pb2.ModelConfig()
        model_config.name = self.model_name
        model_config.platform = _PLATFORM_PER_FORMAT[self.model_format]
        model_config.max_batch_size = self.max_batch_size

        if self.inputs:

            def _rewrite_io_spec(spec_, item):
                dtype = f"TYPE_{client_utils.np_to_triton_dtype(spec_.dtype)}"
                dims = [1] if len(spec_.shape) <= 1 else spec_.shape[1:]  # do not pass batch size

                item.name = spec_.name
                item.dims.extend(dims)
                item.data_type = getattr(grpc_client.model_config_pb2, dtype)
                if len(spec_.shape) <= 1:
                    item.reshape.shape.extend([])

            for spec in self.inputs:
                input_item = model_config.input.add()
                _rewrite_io_spec(spec, input_item)

            for spec in self.outputs:
                output_item = model_config.output.add()
                _rewrite_io_spec(spec, output_item)

        if self.accelerator == Accelerator.TRT:
            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "tensorrt"
            accelerator.parameters["precision_mode"] = self.precision.value.upper()

        elif self.accelerator == Accelerator.AMP:
            accelerator = model_config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "auto_mixed_precision"

        if model_config.platform == "tensorrt_plan":
            model_config.optimization.cuda.graphs = self.capture_cuda_graph

        if any([len(self.preferred_batch_sizes) > 0, self.max_queue_delay_us > 0]):
            model_support_batching = self.max_batch_size > 0
            if model_support_batching:
                model_config.dynamic_batching.max_queue_delay_microseconds = max(int(self.max_queue_delay_us), 0)
                preferred_batch_sizes = self.preferred_batch_sizes or [self.max_batch_size]
                for preferred_batch_size in preferred_batch_sizes:
                    model_config.dynamic_batching.preferred_batch_size.append(int(preferred_batch_size))
            else:
                LOGGER.warning("Ignore dynamic batching parameters as model doesn't support batching")

        instance_group = model_config.instance_group.add()
        instance_group.kind = instance_group.KIND_GPU
        instance_group.count = self.gpu_engine_count

        config_payload = MessageToString(model_config)
        LOGGER.debug(f"Generated Triton config:\n{config_payload}")

        return config_payload

    def save(self, model_dir: Union[str, Path]) -> str:
        """
        Serialize ModelConfig to prototxt and save to model_dir directory.
        model_dir: Union[str, Path] - path to directory where configuration will be stored
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = model_dir / "config.pbtxt"
        config_payload = self.generate_prototxt_payload()

        with config_path.open("w+") as cfg:
            cfg.write(config_payload)

        return config_payload

    @classmethod
    def create(
        cls,
        model_path: Union[str, Path],
        model_name: str,
        model_version: str,
        model_format: str,
        max_batch_size: int,
        precision: str,
        gpu_engine_count: int,
        preferred_batch_sizes: List[int],
        max_queue_delay_us: int,
        capture_cuda_graph: int,
        accelerator: str,
    ):
        model_path = Path(model_path)
        inputs: Optional[List[TensorSpec]] = None
        outputs: Optional[List[TensorSpec]] = None
        if model_format.startswith("ts-"):
            io_spec = _load_io_specs(exported_model=model_path)
            inputs, outputs = list(io_spec.inputs.values()), list(io_spec.outputs.values())
        elif model_format.startswith("onnx"):
            # this is temporary fix for missing auto-generation of model configuration for onnx runtime
            # fixed in xxx version

            model: onnx.ModelProto = OnnxFromPath(model_path.as_posix())()
            inputs = get_input_metadata(model.graph)
            outputs = get_output_metadata(model.graph)
            inputs = [TensorSpec.from_polygraphy_metadata_tuple(name, meta) for name, meta in inputs.items()]
            outputs = [TensorSpec.from_polygraphy_metadata_tuple(name, meta) for name, meta in outputs.items()]

        return ModelConfig(
            model_name=model_name,
            model_version=model_version,
            model_format=Format(model_format),
            max_batch_size=max_batch_size,
            precision=Precision(precision),
            gpu_engine_count=gpu_engine_count,
            preferred_batch_sizes=preferred_batch_sizes,
            max_queue_delay_us=max_queue_delay_us,
            capture_cuda_graph=capture_cuda_graph,
            accelerator=Accelerator(accelerator),
            inputs=inputs,
            outputs=outputs,
        )
