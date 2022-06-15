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
from distutils.version import LooseVersion
from pathlib import Path
from typing import Dict, Optional

import polygraphy
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, Profile, SaveEngine, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc
from polygraphy.logger import G_LOGGER

from model_navigator.converter.config import TensorRTConversionConfig, TensorRTPrecision, TensorRTPrecisionMode
from model_navigator.converter.dataloader import Dataloader
from model_navigator.converter.polygraphy.comparator import ToleranceParameterHelper
from model_navigator.converter.utils import navigator_subprocess, prepare_log_header
from model_navigator.exceptions import ModelNavigatorConverterCommandException, ModelNavigatorConverterException
from model_navigator.model import Format

LOGGER = logging.getLogger("polygraphy.transformers")

POLYGRAPHY_VERSION = LooseVersion(polygraphy.__version__)
DEFAULT_TOLERANCES = {"rtol": 1e-5, "atol": 1e-5}


class LegacyFormatSeparator:
    name_shape_sep = ","
    shape_sep = "x"
    shape_prefix = ""
    shape_suffix = ""


class UpdatedFormatSeparators:
    name_shape_sep = ":"
    shape_sep = ","
    shape_prefix = "["
    shape_suffix = "]"


def _serialize_shapes(shapes):
    new_format = POLYGRAPHY_VERSION > LooseVersion("0.24.2")  # not sure about this version
    sep = UpdatedFormatSeparators if new_format else LegacyFormatSeparator
    return [
        f"{name}{sep.name_shape_sep}{sep.shape_prefix}{sep.shape_sep.join(map(str, shape))}{sep.shape_suffix}"
        for name, shape in shapes.items()
    ]


def _serialize_tolerance(name, value):
    sep = ":" if POLYGRAPHY_VERSION > LooseVersion("0.24.2") else ","
    return f"{name}{sep}{value}" if name else str(value)


def _validate_keys(io_type, expected_io_names, io_names):
    missing_keys = sorted(set(expected_io_names) - set(io_names))
    unknown_keys = sorted(set(io_names) - set(expected_io_names))
    if missing_keys:
        missing_keys = ", ".join(missing_keys)
        raise ModelNavigatorConverterException(f"{io_type} is defined but there are missing keys: {missing_keys}")

    if unknown_keys:
        unknown_keys = ", ".join(unknown_keys)
        expected_io_names = ", ".join(expected_io_names)
        raise ModelNavigatorConverterException(
            f"{io_type} is defined but there are unknown keys: {unknown_keys}; possible keys: {expected_io_names}"
        )


def _run_polygraphy(
    input_path: Path,
    output_path: Path,
    comparator_inputs_path: Path,
    comparator_outputs_path: Path,
    tensorrt_config: TensorRTConversionConfig,
    dataloader: Dataloader,
    rtol: Dict[str, float],
    atol: Dict[str, float],
    trt_precision_flags,
    verbose: bool,
):
    if verbose:
        G_LOGGER.severity = G_LOGGER.VERBOSE
    profile = Profile()
    for name in dataloader.min_shapes:
        profile.add(name, dataloader.min_shapes[name], dataloader.opt_shapes[name], dataloader.max_shapes[name])
    profiles = [profile]
    config = CreateConfig(
        max_workspace_size=tensorrt_config.max_workspace_size,
        profiles=profiles,
        sparse_weights=tensorrt_config.sparse_weights,
        **{precision: True for precision in trt_precision_flags},
    )

    # Loaders
    parse_network_from_onnx = NetworkFromOnnxPath(
        input_path.as_posix(), explicit_precision=tensorrt_config.explicit_precision
    )
    build_engine = EngineFromNetwork(parse_network_from_onnx, config=config)
    save_engine = SaveEngine(build_engine, output_path.as_posix())
    build_onnxrt_session = SessionFromOnnx(input_path.as_posix())

    # Runners
    runners = [
        TrtRunner(save_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    # Runner Execution
    results = Comparator.run(runners, data_loader=dataloader, save_inputs_path=comparator_inputs_path.as_posix())
    results.save(comparator_outputs_path.as_posix())

    success = True
    # Accuracy Comparison
    success &= bool(Comparator.compare_accuracy(results, compare_func=CompareFunc.simple(atol=atol, rtol=rtol)))
    success &= Comparator.validate(results)
    return success


def onnx2trt(
    *,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    tensorrt_config: TensorRTConversionConfig,
    dataloader: Dataloader,
    rtol: Optional[Dict[str, float]] = None,
    atol: Optional[Dict[str, float]] = None,
    verbose: bool = False,
):
    LOGGER.info("Polygraphy onnx2trt started.")

    if tensorrt_config.precision_mode == TensorRTPrecisionMode.HIERARCHY:
        trt_precision_flags = {
            TensorRTPrecision.FP32: ["tf32"],
            TensorRTPrecision.TF32: ["tf32"],
            TensorRTPrecision.FP16: ["tf32", "fp16"],
            TensorRTPrecision.INT8: ["tf32", "fp16", "int8"],
        }[tensorrt_config.precision]
    elif tensorrt_config.precision_mode == TensorRTPrecisionMode.SINGLE:
        trt_precision_flags = {
            TensorRTPrecision.FP32: ["tf32"],
            TensorRTPrecision.TF32: ["tf32"],
            TensorRTPrecision.FP16: ["fp16"],
            TensorRTPrecision.INT8: ["int8"],
        }[tensorrt_config.precision]
    else:
        raise ModelNavigatorConverterException(
            f"Unsupported precision mode {tensorrt_config.precision_mode}. Only {TensorRTPrecisionMode.HIERARCHY} and {TensorRTPrecisionMode.SINGLE} are allowed"
        )

    LOGGER.warning("This conversion should be done on target GPU platform")
    if atol is None:
        atol = {name: DEFAULT_TOLERANCES["atol"] for name in dataloader.max_shapes}
    if rtol is None:
        rtol = {name: DEFAULT_TOLERANCES["rtol"] for name in dataloader.max_shapes}

    comparator_inputs_path = log_path.with_suffix(".comparator_inputs.json")
    comparator_outputs_path = log_path.with_suffix(".comparator_outputs.json")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            prepare_log_header(log_file, Format.TORCHSCRIPT, Format.ONNX)
            with navigator_subprocess(log_file=log_file, verbose=verbose) as navigator:
                success = navigator.module("model_navigator.converter.polygraphy.transformers")._run_polygraphy(
                    input_path,
                    output_path,
                    comparator_inputs_path,
                    comparator_outputs_path,
                    tensorrt_config,
                    dataloader,
                    rtol,
                    atol,
                    trt_precision_flags,
                    verbose,
                )
        LOGGER.info("onnx2trt command succeed.")
    except Exception as e:
        LOGGER.warning(f"Polygraphy onnx2trt conversion failed. Details can be found in logfile: {log_path}")
        raise ModelNavigatorConverterCommandException(message=str(e), log_path=log_path)

    # Report Results
    if not success:
        if comparator_inputs_path.exists() and comparator_outputs_path.exists():
            _dump_tolerace_parameters_if_possible(comparator_inputs_path, comparator_outputs_path)

        msg = f"Polygraphy onnx2trt conversion failed. Details can be found in logfile: {log_path}"
        LOGGER.warning(msg)
        raise ModelNavigatorConverterCommandException(message=msg, log_path=log_path)
    else:
        if not verbose and comparator_inputs_path.exists():
            LOGGER.debug(f"Remove comparator input file {comparator_inputs_path}")
            comparator_inputs_path.unlink()
        if not verbose and comparator_outputs_path.exists():
            LOGGER.debug(f"Remove comparator output file {comparator_outputs_path}")
            comparator_outputs_path.unlink()
        LOGGER.info("Polygraphy onnx2trt succeeded.")


def _dump_tolerace_parameters_if_possible(comparator_inputs_path, comparator_outputs_path):
    from model_navigator.cli.spec import ComparatorConfigCli

    tolerance_helper = ToleranceParameterHelper(comparator_inputs_path, comparator_outputs_path)
    atol, rtol = tolerance_helper.get_tolerance_parameters()
    output_value_ranges = tolerance_helper.get_outputs_value_ranges()

    if atol and rtol:
        atol_cli = " ".join(["--atol"] + ComparatorConfigCli.atol.serialize_default_callback(param=None, value=atol))
        rtol_cli = " ".join(["--rtol"] + ComparatorConfigCli.rtol.serialize_default_callback(param=None, value=rtol))

        value_ranges_str = " ".join(
            ComparatorConfigCli.rtol.serialize_default_callback(param=None, value=output_value_ranges)
        )
        LOGGER.debug(f"Output value ranges: {value_ranges_str}")
        LOGGER.warning(
            "For data which was used during conversion verification, "
            f"tolerance parameters which make conversion correctness will pass: {atol_cli} {rtol_cli}. "
            "There is no warranty that this parameter make sense. They require verification!"
        )
