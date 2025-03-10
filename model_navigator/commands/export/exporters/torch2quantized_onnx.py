# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
"""Export PyTorch model to quantized ONNX using ModelOpt."""

import inspect
import pathlib
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Union

import fire
import modelopt.torch.quantization as mtq
import torch  # pytype: disable=import-error # noqa: F401
from modelopt.core.torch.quantization.config import NVFP4_FP8_MHA_CONFIG

from model_navigator.core.dataloader import load_samples
from model_navigator.core.logger import LOGGER
from model_navigator.core.tensor import TensorMetadata
from model_navigator.utils.common import numpy_to_torch_dtype


def get_model() -> torch.nn.Module:
    """Get model instance.

    Returns:
        Model to be exported.
    """
    raise NotImplementedError("Please implement the get_model() function to reproduce the export error.")


def _configure_quantizers_for_onnx_export(model, verbose=False):
    """Configure quantizers in the model for ONNX export.

    Sets appropriate attributes on input and weight quantizers to ensure proper ONNX export.
    For input quantizers, sets dynamic quantization mode and Half precision.
    For weight quantizers, sets static quantization mode.

    Based on ModelOpt example: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/diffusers/quantization/onnx_utils/export.py#L192

    Args:
        model: The quantized PyTorch model with quantizers to configure
        verbose: Whether to log detailed information about each configured quantizer
    """
    LOGGER.info("Configuring quantizers for ONNX export")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            if hasattr(module, "input_quantizer") and module.input_quantizer is not None:
                if verbose:
                    LOGGER.info(f"Setting quantizer attributes for {name}.input_quantizer")
                module.input_quantizer._onnx_quantizer_type = "dynamic"
                module.input_quantizer._trt_high_precision_dtype = "Half"
            if hasattr(module, "weight_quantizer") and module.weight_quantizer is not None:
                if verbose:
                    LOGGER.info(f"Setting quantizer attributes for {name}.weight_quantizer")
                module.weight_quantizer._onnx_quantizer_type = "static"


def export(
    exported_model_path: str,
    opset: int,
    input_metadata: Dict[str, Any],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Union[Dict[int, str], List[int]]]],
    batch_dim: Optional[int],
    target_device: str,
    precision: str,
    custom_args: Dict[str, Any],
    navigator_workspace: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Export PyTorch model to quantized ONNX using ModelOpt.

    Args:
        exported_model_path (str): Output ONNX model path.
        opset (int): ONNX opset.
        input_metadata (Dict): Model input metadata.
        output_names (List[str]): List of model output names.
        dynamic_axes (Optional[Dict[str, Union[Dict[int, str], List[int]]]]): Configuration of dynamic axes.
        batch_dim (Optional[int]): Batch dimension.
        target_device (str): Device to run on.
        precision (str): Precision type for quantization.
        custom_args (Dict[str, Any]): Passthrough parameters for torch.onnx.export.
        navigator_workspace (Optional[str]): Model Navigator workspace path.
        verbose (bool): Enable verbose logging.
    """
    if not navigator_workspace:
        navigator_workspace = pathlib.Path.cwd()
    navigator_workspace = pathlib.Path(navigator_workspace)

    # Get original model
    original_model = get_model()

    # Create a deep copy for quantization
    LOGGER.info("Creating a deep copy of the model for quantization")
    model_copy = deepcopy(original_model)

    # Offload original model to CPU
    original_model.to("cpu")

    try:
        # Move model copy to target device
        model_copy = model_copy.to(target_device)

        # Load calibration samples
        LOGGER.info("Loading calibration samples")
        correctness_samples = load_samples("correctness_samples", navigator_workspace, batch_dim)
        if not correctness_samples:
            LOGGER.error("No correctness samples found for calibration")
            raise RuntimeError("No calibration samples found")

        # Convert samples to PyTorch tensors
        torch_samples = []
        for sample in correctness_samples:
            sample_dict = {}
            for name, tensor in sample.items():
                torch_sample = torch.from_numpy(tensor)
                torch_sample = torch_sample.to(target_device)
                sample_dict[name] = torch_sample
            torch_samples.append(sample_dict)

        calibration_data = [list(sample.values()) for sample in torch_samples]

        # Define calibration function
        def forward_loop(model):
            for sample in calibration_data:
                model(*sample)

        LOGGER.info("Using NVFP4_FP8_MHA_CONFIG quantization config for precision NVFP4")

        # Run quantization
        LOGGER.info("Starting model quantization (this may take several minutes)...")
        quantized_model = mtq.quantize(
            model_copy,
            NVFP4_FP8_MHA_CONFIG,
            forward_loop,
        )

        LOGGER.info("Model quantization completed")

        # Prepare input for ONNX export
        input_metadata = TensorMetadata.from_json(input_metadata)
        correct_sample = correctness_samples[0]
        dummy_input = {n: torch.from_numpy(val).to(target_device) for n, val in correct_sample.items()}

        # Adjust input dtypes to match input_metadata
        for n, spec in input_metadata.items():
            if not isinstance(spec.dtype, torch.dtype):
                torch_dtype = numpy_to_torch_dtype(spec.dtype)
            else:
                torch_dtype = spec.dtype
            dummy_input[n] = dummy_input[n].to(torch_dtype).to(target_device)

        dummy_input = input_metadata.unflatten_sample(dummy_input)

        # torch.onnx.export requires inputs to be a tuple or tensor
        if isinstance(dummy_input, Mapping):
            dummy_input = (dummy_input,)

        # Get expected function signature for forward method
        forward_argspec = inspect.getfullargspec(model_copy.forward)
        forward_args = forward_argspec.args[1:]  # Skip 'self'

        # Create input_names for ONNX model
        args_mapping, kwargs_mapping = input_metadata.pytree_metadata.get_names_mapping()

        input_names = []
        for args_names in args_mapping:
            input_names.extend(args_names)

        for argname in forward_args:
            if argname in kwargs_mapping:
                input_names.extend(kwargs_mapping[argname])
        # Configure quantizers for ONNX export
        _configure_quantizers_for_onnx_export(quantized_model, verbose=verbose)

        # Export to ONNX
        exported_model_path = pathlib.Path(exported_model_path)
        if not exported_model_path.is_absolute():
            exported_model_path = navigator_workspace / exported_model_path

        LOGGER.info(f"Exporting quantized model to ONNX at {exported_model_path}")
        torch.onnx.export(
            quantized_model,
            args=dummy_input,
            f=exported_model_path.as_posix(),
            verbose=verbose,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **custom_args,
        )

        # Apply post-processing to the ONNX model to optimize for TensorRT
        try:
            LOGGER.info("Applying fp4qdq_to_2dq post-processing to the exported ONNX model")
            import onnx
            from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

            # Load the exported ONNX model
            onnx_model = onnx.load(exported_model_path)

            # Apply fp4qdq_to_2dq transformation
            optimized_model = fp4qdq_to_2dq(onnx_model)

            # Save the optimized model back to the same path
            onnx.save(optimized_model, exported_model_path)
            LOGGER.info("Post-processing completed successfully")
        except ImportError as e:
            LOGGER.warning(f"Could not import modelopt.onnx.quantization.qdq_utils: {str(e)}")
            LOGGER.warning("Skipping fp4qdq_to_2dq post-processing")
        except Exception as e:
            LOGGER.warning(f"Error during fp4qdq_to_2dq post-processing: {str(e)}")
            LOGGER.warning("Using the original exported ONNX model without post-processing")

        LOGGER.info("Quantized ONNX export completed successfully")

        # Clean up
        del model_copy
        del quantized_model
        torch.cuda.empty_cache()

    except Exception as e:
        LOGGER.error(f"Error during quantized ONNX export: {str(e)}")
        raise


if __name__ == "__main__":
    fire.Fire(export)
