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
import dataclasses
import json
import subprocess
import typing
from collections import OrderedDict

import numpy as np
from polygraphy.backend.trt import Profile
from polygraphy.backend.trt import TrtRunner as _TrtRunner
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy_trtexec.backend import TrtexecRunner as _TrtexecRunner

from model_navigator.framework_api.common import TensorMetadata
from model_navigator.framework_api.logger import LOGGER
from model_navigator.framework_api.runners.base import INavigatorRunner, INavigatorStabilizedRunner
from model_navigator.model import Format
from model_navigator.utils.config import dataclass2dict


class TrtRunner(INavigatorRunner, _TrtRunner):
    """
    Runs inference using PyTorch.
    """

    # TODO: observe TRT API on this cast
    trt_casts = {np.dtype(np.int64): np.int32}

    def infer(self, feed_dict, check_inputs=None, *args, **kwargs):
        feed_dict = {
            name: self._cast_tensor(tensor) for name, tensor in feed_dict.items() if name in self.get_input_metadata()
        }
        return super().infer(feed_dict, check_inputs, *args, **kwargs)

    def _cast_tensor(self, tensor):
        if tensor.dtype in self.trt_casts:
            LOGGER.debug(f"Casting f{tensor.dtype} tensor to f{self.trt_casts[tensor.dtype]}.")
            return tensor.astype(self.trt_casts[tensor.dtype])
        return tensor


@dataclasses.dataclass
class TrtExecRuntimeConfig:
    # see trtexec --help for meaning and default of below values
    use_cuda_graph: typing.Optional[bool] = None
    avg_runs: typing.Optional[int] = None
    best: typing.Optional[bool] = None
    duration: typing.Optional[int] = None
    device: typing.Optional[int] = None
    streams: typing.Optional[int] = None
    min_timing: typing.Optional[int] = None
    avg_timing: typing.Optional[int] = None
    expose_dma: typing.Optional[bool] = None
    no_data_transfers: typing.Optional[bool] = None
    warmup: typing.Optional[int] = None
    iterations: typing.Optional[int] = None
    use_spin_wait: typing.Optional[bool] = None
    threads: typing.Optional[bool] = None
    use_managed_memory: typing.Optional[bool] = None
    dump_refit: typing.Optional[bool] = None
    dump_output: typing.Optional[bool] = None
    dump_profile: typing.Optional[bool] = None
    dump_layer_info: typing.Optional[bool] = None
    refit: typing.Optional[bool] = None
    separate_profile_run: typing.Optional[bool] = None
    no_builder_cache: typing.Optional[bool] = None
    # one of layer_names_only|detailed|none
    profiling_verbosity: typing.Optional[str] = None
    # key is layer name, "*" can be used as a layer name to specify the default precision for all the unspecified layers
    # value is precision: "fp32"|"fp16"|"int32"|"int8"
    layer_precisions: typing.Optional[typing.Dict[str, str]] = None
    # key is layer name, "*" can be used as a layer name to specify the default precision for all the unspecified layers
    # value is output type: "fp32"|"fp16"|"int32"|"int8"["+"type]
    # If a layer has more than one output, then multiple types separated by "+" can be provided for this layer.
    layer_output_types: typing.Optional[typing.Dict[str, str]] = None
    use_dla_core: typing.Optional[str] = None
    tf32: typing.Optional[bool] = None
    fp16: typing.Optional[bool] = None
    int8: typing.Optional[bool] = None
    allow_gpu_fallback: typing.Optional[bool] = None
    # one of prefer|obey|none
    precision_constraints: typing.Optional[str] = None
    # bytes with optional `K`, `M`, or `G` suffixes
    workspace: typing.Optional[str] = None
    use_dla: typing.Optional[bool] = None
    plugins: typing.Optional[typing.List[str]] = None
    save_engine: typing.Optional[str] = None


def _format_dict(d: typing.Dict):
    return ",".join([f"{k}:{v}" for k, v in d.items()])


class TrtexecRunner(INavigatorStabilizedRunner, _TrtexecRunner):
    # TODO: observe TRT API on this cast
    trt_casts = {np.dtype(np.int64): np.int32}

    def __init__(
        self,
        model,
        model_format: Format,
        *,
        input_metadata: typing.Optional[TensorMetadata] = None,
        input_profile: typing.Optional[Profile] = None,
        runtime_config: typing.Optional[TrtExecRuntimeConfig] = None,
    ):

        model_path = model
        model_type = {model_format.ONNX: "onnx", model_format.TENSORRT: "engine"}[model_format]

        kwargs = {
            "model_path": model_path,
            "model_type": model_type,
        }
        if input_metadata:
            kwargs["input_shapes"] = input_metadata
        if input_profile:
            kwargs["profile_dicts"] = input_profile

        if runtime_config:
            runtime_config_dict = dataclass2dict(runtime_config)
            runtime_config_dict["trtexec_warmup"] = runtime_config_dict.pop("warmup")
            runtime_config_dict["trtexec_iterations"] = runtime_config_dict.pop("iterations")
            runtime_config_dict["trtexec_no_builder_cache"] = runtime_config_dict.pop("no_builder_cache")
            runtime_config_dict["trtexec_profiling_verbosity"] = runtime_config_dict.pop("profiling_verbosity")

            if runtime_config_dict["layer_precisions"]:
                runtime_config_dict["layer_precisions"] = _format_dict(runtime_config_dict["layer_precisions"])
            if runtime_config_dict["layer_output_types"]:
                runtime_config_dict["layer_output_types"] = _format_dict(runtime_config_dict["layer_output_types"])

            kwargs.update(**runtime_config_dict)

        self._request_count = None
        self._avg_latency = None
        self._p50_latency = None
        self._p99_latency = None

        super().__init__(**kwargs)

    def infer_impl(self, feed_dict):
        # TODO: This is WAR for TRTExec Runner - remove once implemented in source runner
        # Adds other args that need to generated during inference. For example,
        # `feed_dict` is used to generate the args for `loadInputs`
        self.construct_final_cmd(feed_dict)
        G_LOGGER.info(f"The trtexec command being run: {self.cmd_args}")
        perf_output = subprocess.run(self.cmd_args, stdout=subprocess.PIPE, text=True).stdout
        inference_time_stats = self._get_inference_stats(perf_output)
        G_LOGGER.verbose(f"Inference time statistics: {inference_time_stats}")

        outputs = self.read_output_file()

        self._request_count = self._get_request_count(perf_output)
        self._avg_latency = inference_time_stats["mean"]
        self._p50_latency = inference_time_stats["median"]
        self._p99_latency = inference_time_stats["percentile(99%)"]

        # inference_time_stats records time in 'ms'. However, polygraphy
        # expects time in seconds.
        self.inference_time = self._avg_latency / 1000.0

        return outputs

    def infer(self, feed_dict, check_inputs=None, *args, **kwargs):
        feed_dict = {
            name: self._cast_tensor(tensor) for name, tensor in feed_dict.items() if name in self.get_input_metadata()
        }

        # need to move this part to polygraphy_trtexec
        input_shapes = {name: tensor.shape for name, tensor in feed_dict.items()}
        input_shapes = {name: "x".join(map(str, tensor_shape)) for name, tensor_shape in input_shapes.items()}
        input_shapes = _format_dict(input_shapes)
        cmd_args_without_input_shapes_and_load_inputs = [
            item for item in self.cmd_args if not item.startswith("--shapes=") and not item.startswith("--loadInputs=")
        ]
        self.cmd_args = cmd_args_without_input_shapes_and_load_inputs
        self.add_cmd_args("shapes", input_shapes)

        return super().infer(feed_dict, check_inputs, *args, **kwargs)

    def read_output_file(self):
        """
        Reads the output from the output file generated by the trtexec binary
        """

        # need to move this part to polygraphy_trtexec
        outputs = OrderedDict()
        with open(self.export_output_file.name) as export_file:
            content = json.load(export_file)
        for entry in content:
            name, dimensions, values = entry["name"], entry["dimensions"], entry["values"]
            dimensions = [int(d) for d in dimensions.split("x")]
            outputs[name] = np.array(values).reshape(*dimensions)
        return outputs

    def avg_latency(self):
        """
        Returns the mean inference time required during the last call to ``infer()``.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        if self._avg_latency is None:
            G_LOGGER.warning(
                "{:35} | avg_latency was not set. Inference time will be incorrect!"
                "To correctly compare runtimes, please set the avg_latency property in the"
                "infer() function".format(self.name),
                mode=LogMode.ONCE,
            )
            return None
        return self._avg_latency

    def std_latency(self):
        """
        Returns the std of measured stable latency.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        # TODO: Obtain from TRTExec once available
        return 0.0

    def p50_latency(self):
        """
        Returns the p50 of measured stable latency.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        if self._p50_latency is None:
            G_LOGGER.warning(
                "{:35} | p50_latency was not set. Inference time will be incorrect!"
                "To correctly compare runtimes, please set the p50_latency property in the"
                "infer() function".format(self.name),
                mode=LogMode.ONCE,
            )
            return None
        return self._p50_latency

    def p90_latency(self):
        """
        Returns the p90 of measured stable latency.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        # TODO: Obtain from TRTExec once available
        return 0.0

    def p95_latency(self):
        """
        Returns the p95 of measured stable latency.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        # TODO: Obtain from TRTExec once available
        return 0.0

    def p99_latency(self):
        """
        Returns the p99 of measured stable latency.

        Returns:
            float: The time in milliseconds, or None if runtime was not measured by the runner.
        """
        if self._p99_latency is None:
            G_LOGGER.warning(
                "{:35} | p99_latency was not set. Inference time will be incorrect!"
                "To correctly compare runtimes, please set the p99_latency property in the"
                "infer() function".format(self.name),
                mode=LogMode.ONCE,
            )
            return None
        return self._p99_latency

    def request_count(self):
        """
        Returns the number of queries of measurement.

        Returns:
            float: The number of queries, or None if runtime was not measured by the runner.
        """
        if self._request_count is None:
            G_LOGGER.warning(
                "{:35} | request_count was not set. Inference time will be incorrect!"
                "To correctly compare runtimes, please set the request_count property in the"
                "infer() function".format(self.name),
                mode=LogMode.ONCE,
            )
            return None
        return self._request_count

    def _cast_tensor(self, tensor):
        if tensor.dtype in self.trt_casts:
            LOGGER.debug(f"Casting f{tensor.dtype} tensor to f{self.trt_casts[tensor.dtype]}.")
            return tensor.astype(self.trt_casts[tensor.dtype])
        return tensor

    def _get_inference_stats(self, perf_output):
        """
        Reads the output from the performance summary generated by the trtexec
        binary to extract the required performance statistics
        """
        inference_time_stats = {}
        inference_time_field = "Latency:"
        for line in perf_output.split("\n"):
            index = line.find(inference_time_field)
            if index >= 0:
                stats = line[index + len(inference_time_field) :].split(",")
                for stat in stats:
                    metric, value = stat.split("=")
                    value = value.strip().split(" ")[0]
                    inference_time_stats[metric.strip()] = float(value)
                return inference_time_stats
        G_LOGGER.critical(
            "Could not read inference time for trtexec backend. This " "might cause polygraphy to misbehave"
        )

    @staticmethod
    def _get_request_count(perf_output):
        """
        Reads the output from the performance summary generated by the trtexec
        binary to extract the required performance statistics
        """
        inference_time_field = "Timing trace has"
        for line in perf_output.split("\n"):
            index = line.find(inference_time_field)
            if index >= 0:
                stats = line[index + len(inference_time_field) :].split(" ")
                request_count = int(stats[1])
                return request_count
        G_LOGGER.critical(
            "Could not read request count for trtexec backend. This " "might cause polygraphy to misbehave"
        )
