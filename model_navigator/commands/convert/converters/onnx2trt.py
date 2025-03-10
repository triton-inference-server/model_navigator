# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Script for ONNX to TensorRT conversion."""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import fire
from polygraphy import mod
from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine

from model_navigator.configuration import PrecisionType, TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.core.dataloader import load_samples
from model_navigator.core.logger import LOGGER
from model_navigator.frameworks import is_modelopt_available
from model_navigator.frameworks.tensorrt.timing_tactics import TimingCacheManager, trt_cache_inplace_cache_dir

trt = mod.lazy_import("tensorrt")


def _get_precisions(precision, precision_mode):
    precision = TensorRTPrecision(precision)
    precision_mode = TensorRTPrecisionMode(precision_mode)
    if precision_mode == TensorRTPrecisionMode.HIERARCHY:
        tf32, fp16, bf16, fp8, int8, nvfp4 = {
            # TODO: Enable hierarchical BF16 for FP16, FP8 and INT8 after it's supported
            TensorRTPrecision.FP32: [True, False, False, False, False, False],
            # TensorRTPrecision.FP16: [True, True, True, False, False],
            TensorRTPrecision.FP16: [True, True, False, False, False, False],
            TensorRTPrecision.BF16: [True, True, True, False, False, False],
            # TensorRTPrecision.FP8: [True, True, True, True, False],
            TensorRTPrecision.FP8: [True, True, False, True, False, False],
            # TensorRTPrecision.INT8: [True, True, True, False, True],
            TensorRTPrecision.INT8: [True, True, False, False, True, False],
            TensorRTPrecision.NVFP4: [True, True, False, False, False, True],
        }[precision]
    elif precision_mode == TensorRTPrecisionMode.SINGLE:
        tf32, fp16, bf16, fp8, int8, nvfp4 = {
            TensorRTPrecision.FP32: [True, False, False, False, False, False],
            TensorRTPrecision.FP16: [False, True, False, False, False, False],
            TensorRTPrecision.BF16: [False, False, True, False, False, False],
            TensorRTPrecision.FP8: [False, False, False, True, False, False],
            TensorRTPrecision.INT8: [False, False, False, False, True, False],
            TensorRTPrecision.NVFP4: [False, False, False, False, False, True],
        }[precision]
    else:
        raise ValueError(
            f"Unsupported precision mode {precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and "
            f"{TensorRTPrecisionMode.SINGLE} are allowed"
        )
    return tf32, fp16, bf16, fp8, int8, nvfp4


def convert(
    exported_model_path: str,
    converted_model_path: str,
    profiles: List[Dict[str, Dict[str, Tuple[int, ...]]]],
    max_workspace_size: int,
    precision: str,
    precision_mode: str,
    optimization_level: Optional[int] = None,
    compatibility_level: Optional[str] = None,
    navigator_workspace: Optional[str] = None,
    onnx_parser_flags: Optional[List[int]] = None,
    timing_cache_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    custom_args: Optional[Dict[str, Any]] = None,
    batch_dim: Optional[int] = None,
    model_precision: Optional[PrecisionType] = None,
    quantized_onnx_path: Optional[str] = None,
) -> None:
    """Run conversion from ONNX to TensorRT.

    Args:
        exported_model_path: ONNX model path.
        converted_model_path: Output TensorRT engine path.
        profiles: Dictionary with min, opt, max shapes of the inputs.
            The key is an input name and the value is a dictionary with keys ("min", "opt", "max")
            and respective values.
        max_workspace_size: Maximum workspace size in bytes.
        precision: TensorRT precision. Could be "fp16", "fp32", etc.
        precision_mode: TensorRT precision mode.
        navigator_workspace: Model Navigator workspace path.
            When None use current workdir. Defaults to None.
        optimization_level: Optimization level for TensorRT engine
        compatibility_level: Hardware compatibility level for generated engine
        onnx_parser_flags: List of flags to set ONNX parser behavior.
        timing_cache_dir: Directory to save timing cache. Defaults to None which means it will be saved in workspace root.
        model_name: Model name for the timing cache. Defaults to None which means it will be named after the model file.
        custom_args: Dictionary with passthrough parameters.
            For available arguments check PyTorch documentation: https://pytorch.org/TensorRT/py_api/torch_tensorrt.html
        batch_dim: Batch dimension. Defaults to None.
        model_precision: Source model precision.
        quantized_onnx_path: Path to the quantized ONNX model.
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    exported_model_path = pathlib.Path(exported_model_path)
    if not exported_model_path.is_absolute():
        exported_model_path = navigator_workspace / exported_model_path
    exported_model_path = exported_model_path.as_posix()

    if model_name is None:
        model_name = navigator_workspace.stem

    converted_model_path = pathlib.Path(converted_model_path)
    if not converted_model_path.is_absolute():
        converted_model_path = navigator_workspace / converted_model_path
    converted_model_path = converted_model_path.as_posix()

    if quantized_onnx_path:
        quantized_onnx_path = pathlib.Path(quantized_onnx_path)
        if not quantized_onnx_path.is_absolute():
            quantized_onnx_path = navigator_workspace / quantized_onnx_path
        quantized_onnx_path = quantized_onnx_path.as_posix()

    custom_args = custom_args or {}

    trt_profiles = []
    for profile in profiles:
        trt_profile = Profile()
        for name, dims in profile.items():
            trt_profile.add(name, **dims)
        trt_profiles.append(trt_profile)
    if not trt_profiles:
        trt_profiles = [Profile()]

    # nvfp4 is currently not used as flag for converter, skip it
    tf32, fp16, bf16, fp8, int8, _ = _get_precisions(precision, precision_mode)

    # Determine the path to use for ONNX model
    onnx_path = exported_model_path

    # Check if we need to perform quantization
    should_quantize = (
        model_precision not in ("fp8", "int8")
        and is_modelopt_available()
        and TensorRTPrecision(precision) in (TensorRTPrecision.FP8, TensorRTPrecision.INT8)
    )

    # Use ModelOpt for quantization if needed
    if quantized_onnx_path and should_quantize:
        LOGGER.info("Quantize model through TensorRT ModelOpt with {} precision", precision)
        import modelopt.onnx.quantization as moq  # pytype: disable=import-error # noqa: F401

        correctness_samples = load_samples("correctness_samples", navigator_workspace, batch_dim)
        calibration_data = {name: tensor for sample in correctness_samples for name, tensor in sample.items()}

        # Prepare quantization parameters
        quantize_kwargs = {
            "onnx_path": onnx_path,
            "calibration_data": calibration_data,
            "output_path": quantized_onnx_path,
            "quantize_mode": precision,
        }

        moq.quantize(**quantize_kwargs)
        LOGGER.info("Quantized ONNX model saved in {}", quantized_onnx_path)
        onnx_path = quantized_onnx_path
    # For NVFP4, always use the quantized path (even if not quantized yet)
    elif quantized_onnx_path and TensorRTPrecision(precision) == TensorRTPrecision.NVFP4:
        onnx_path = quantized_onnx_path

    if TensorRTPrecision(precision) == TensorRTPrecision.NVFP4:
        strongly_typed = True
    else:
        strongly_typed = False
    network = network_from_onnx_path(onnx_path, flags=onnx_parser_flags, strongly_typed=strongly_typed)

    config_kwargs = {}
    if optimization_level:
        config_kwargs["builder_optimization_level"] = optimization_level
    if compatibility_level:
        config_kwargs["hardware_compatibility_level"] = compatibility_level

    if max_workspace_size:
        config_kwargs["memory_pool_limits"] = {
            trt.MemoryPoolType.WORKSPACE: max_workspace_size,
        }

    # saving timing cache in model_navigator workspace or ...
    timing_cache = trt_cache_inplace_cache_dir()
    if timing_cache_dir is not None:
        timing_cache = pathlib.Path(timing_cache_dir)

    with TimingCacheManager(model_name=model_name, cache_path=timing_cache) as timing_cache:
        timing_cache = timing_cache.as_posix() if timing_cache else None

        engine = engine_from_network(
            network,
            config=CreateConfig(
                tf32=tf32,
                fp16=fp16,
                bf16=bf16,
                fp8=fp8,
                int8=int8,
                profiles=trt_profiles,
                load_timing_cache=timing_cache,
                **config_kwargs,
                **custom_args,
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=converted_model_path)


if __name__ == "__main__":
    fire.Fire(convert)
