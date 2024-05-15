# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
import itertools

from model_navigator.configuration import (
    Format,
    JitType,
    OnnxConfig,
    TensorFlowConfig,
    TensorFlowTensorRTConfig,
    TensorRTCompatibilityLevel,
    TensorRTConfig,
    TensorRTPrecision,
    TensorRTPrecisionMode,
    TensorRTProfile,
    TorchConfig,
    TorchExportConfig,
    TorchScriptConfig,
    TorchTensorRTConfig,
)
from model_navigator.configuration.model import model_config
from model_navigator.configuration.model.model_config_builder import ModelConfigBuilder
from model_navigator.frameworks import Framework


def test_get_source_torch_config_returns_model_configs_matching_custom_config():
    torch_config = TorchConfig()
    model_configs = {Format.TORCH: []}
    custom_configs = [torch_config]
    ModelConfigBuilder().get_source_torch_config(custom_configs=custom_configs, model_configs=model_configs)

    assert len(model_configs[Format.TORCH]) == 1
    model_configuration = model_configs[Format.TORCH][0]
    assert isinstance(model_configuration, model_config.TorchModelConfig)
    assert model_configuration.parent_key is None

    assert model_configuration.runner_config.autocast is False
    assert model_configuration.runner_config.inference_mode is True
    assert model_configuration.runner_config.device is None


def test_get_source_torch_config_returns_model_configs_matching_custom_config_when_overriden_arguments():
    torch_config = TorchConfig(autocast=True, inference_mode=False, device="cpu")
    model_configs = {Format.TORCH: []}
    custom_configs = [torch_config]
    ModelConfigBuilder().get_source_torch_config(custom_configs=custom_configs, model_configs=model_configs)

    assert len(model_configs[Format.TORCH]) == 1
    model_configuration = model_configs[Format.TORCH][0]
    assert isinstance(model_configuration, model_config.TorchModelConfig)
    assert model_configuration.parent_key is None

    assert model_configuration.runner_config.autocast is True
    assert model_configuration.runner_config.inference_mode is False
    assert model_configuration.runner_config.device == "cpu"


def test_get_source_tensorflow_config_returns_model_configs_matching_custom_config():
    model_configs = {Format.TENSORFLOW: []}
    ModelConfigBuilder().get_source_tensorflow_config(model_configs)

    assert len(model_configs[Format.TENSORFLOW]) == 1
    model_configuration = model_configs[Format.TENSORFLOW][0]
    assert isinstance(model_configuration, model_config.TensorFlowModelConfig)
    assert model_configuration.parent_key is None


def test_get_source_jax_config_returns_model_configs_matching_custom_config():
    model_configs = {Format.JAX: []}
    ModelConfigBuilder().get_source_jax_config(model_configs)

    assert len(model_configs[Format.JAX]) == 1
    model_configuration = model_configs[Format.JAX][0]
    assert isinstance(model_configuration, model_config.JAXModelConfig)
    assert model_configuration.parent_key is None


def test_get_torchscript_config_returns_model_configs_matching_custom_config():
    torch_script_config = TorchScriptConfig()
    model_configs = {Format.TORCHSCRIPT: []}
    custom_configs = [torch_script_config]
    ModelConfigBuilder().get_torchscript_config(custom_configs, model_configs)

    assert len(model_configs[Format.TORCHSCRIPT]) == 2
    for jit_type, model_configuration in zip(torch_script_config.jit_type, model_configs[Format.TORCHSCRIPT]):
        assert isinstance(model_configuration, model_config.TorchScriptConfig)
        assert model_configuration.jit_type == jit_type
        assert model_configuration.format == torch_script_config.format
        assert model_configuration.parent_key is None

        assert model_configuration.runner_config.autocast is False
        assert model_configuration.runner_config.inference_mode is True
        assert model_configuration.runner_config.device is None


def test_get_torchscript_config_returns_model_configs_matching_custom_config_when_overriden_arguments():
    torch_script_config = TorchScriptConfig(
        jit_type=(JitType.TRACE,),
        autocast=True,
        inference_mode=False,
        device="cpu",
    )
    model_configs = {Format.TORCHSCRIPT: []}
    custom_configs = [torch_script_config]
    ModelConfigBuilder().get_torchscript_config(custom_configs, model_configs)

    assert len(model_configs[Format.TORCHSCRIPT]) == 1
    model_configuration = model_configs[Format.TORCHSCRIPT][0]

    assert isinstance(model_configuration, model_config.TorchScriptConfig)
    assert model_configuration.jit_type == JitType.TRACE
    assert model_configuration.format == torch_script_config.format
    assert model_configuration.parent_key is None

    assert model_configuration.runner_config.autocast is True
    assert model_configuration.runner_config.inference_mode is False
    assert model_configuration.runner_config.device == "cpu"


def test_get_torch_export_config_returns_model_configs_matching_custom_config():
    torch_export_config = TorchExportConfig()
    model_configs = {Format.TORCH_EXPORTEDPROGRAM: []}
    custom_configs = [torch_export_config]
    ModelConfigBuilder().get_torch_exportedprogram_config(custom_configs, model_configs)

    assert len(model_configs[Format.TORCH_EXPORTEDPROGRAM]) == 1
    model_configuration = model_configs[Format.TORCH_EXPORTEDPROGRAM][0]
    assert isinstance(model_configuration, model_config.TorchExportedProgram)
    assert model_configuration.parent_key is None

    assert model_configuration.runner_config.autocast is False
    assert model_configuration.runner_config.inference_mode is True
    assert model_configuration.runner_config.device is None


def test_get_torch_export_config_returns_model_configs_matching_custom_config_when_overriden_arguments():
    torch_export_config = TorchExportConfig(autocast=True, inference_mode=False, device="cpu")
    model_configs = {Format.TORCH_EXPORTEDPROGRAM: []}
    custom_configs = [torch_export_config]
    ModelConfigBuilder().get_torch_exportedprogram_config(custom_configs, model_configs)

    assert len(model_configs[Format.TORCH_EXPORTEDPROGRAM]) == 1
    model_configuration = model_configs[Format.TORCH_EXPORTEDPROGRAM][0]
    assert isinstance(model_configuration, model_config.TorchExportedProgram)
    assert model_configuration.parent_key is None

    assert model_configuration.runner_config.autocast is True
    assert model_configuration.runner_config.inference_mode is False
    assert model_configuration.runner_config.device == "cpu"


def test_get_torch_trt_config_returns_model_configs_matching_custom_config():
    torch_config = TorchConfig()
    torch_trt_config = TorchTensorRTConfig(
        precision=(TensorRTPrecision.FP16,),
        precision_mode=TensorRTPrecisionMode.MIXED,
        max_workspace_size=10,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
    )

    custom_configs = [torch_config, torch_trt_config]
    model_configs = {Format.TORCHSCRIPT: [], Format.TORCH_TRT: []}
    ModelConfigBuilder().get_torchscript_config(custom_configs, model_configs)
    ModelConfigBuilder().get_torch_trt_config(custom_configs, model_configs)

    assert len(model_configs[Format.TORCH_TRT]) == 2
    for (torch_model_configuration, precision), torch_trt_model_configuration in zip(
        itertools.product(model_configs[Format.TORCHSCRIPT], torch_trt_config.precision),
        model_configs[Format.TORCH_TRT],
    ):
        assert isinstance(torch_trt_model_configuration, model_config.TorchTensorRTConfig)
        assert isinstance(torch_model_configuration, model_config.TorchScriptConfig)
        assert torch_trt_model_configuration.precision == precision
        assert torch_trt_model_configuration.max_workspace_size == torch_trt_config.max_workspace_size
        assert torch_trt_model_configuration.precision_mode == torch_trt_config.precision_mode
        assert torch_trt_model_configuration.format == torch_trt_config.format
        assert torch_trt_model_configuration.trt_profiles == torch_trt_config.trt_profiles
        assert torch_trt_model_configuration.parent_key == torch_model_configuration.key
        assert torch_trt_model_configuration.runner_config.device is None


def test_get_savedmodel_config_returns_model_configs_matching_custom_config():
    savedmodel_config = TensorFlowConfig(jit_compile=(True, False), enable_xla=(True, False))

    custom_configs = [savedmodel_config]
    model_configs = {Format.TF_SAVEDMODEL: []}
    ModelConfigBuilder().get_savedmodel_config(custom_configs, model_configs)

    assert len(model_configs[Format.TF_SAVEDMODEL]) == 4
    for (jit_compile_option, enable_xla_option), savedmodel_model_config in zip(
        itertools.product(savedmodel_config.jit_compile, savedmodel_config.enable_xla),
        model_configs[Format.TF_SAVEDMODEL],
    ):
        assert isinstance(savedmodel_model_config, model_config.TensorFlowSavedModelConfig)
        assert savedmodel_model_config.jit_compile == jit_compile_option
        assert savedmodel_model_config.enable_xla == enable_xla_option
        assert savedmodel_model_config.format == savedmodel_config.format


def test_get_tf_trt_config_returns_model_configs_matching_custom_config():
    savedmodel_config = TensorFlowConfig()
    tensorflow_tensorrt_config = TensorFlowTensorRTConfig(
        precision=(TensorRTPrecision.FP16,),
        max_workspace_size=10,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
        minimum_segment_size=13,
    )

    custom_configs = [savedmodel_config, tensorflow_tensorrt_config]
    model_configs = {Format.TF_SAVEDMODEL: [], Format.TF_TRT: []}
    ModelConfigBuilder().get_savedmodel_config(custom_configs, model_configs)
    ModelConfigBuilder().get_tf_trt_config(custom_configs, model_configs)

    assert len(model_configs[Format.TF_TRT]) == 1
    for (sm_model_configuration, precision), tf_trt_model_configuration in zip(
        itertools.product(model_configs[Format.TF_SAVEDMODEL], tensorflow_tensorrt_config.precision),
        model_configs[Format.TF_TRT],
    ):
        assert isinstance(tf_trt_model_configuration, model_config.TensorFlowTensorRTConfig)
        assert isinstance(sm_model_configuration, model_config.TensorFlowSavedModelConfig)
        assert tf_trt_model_configuration.precision == precision
        assert tf_trt_model_configuration.max_workspace_size == tensorflow_tensorrt_config.max_workspace_size
        assert tf_trt_model_configuration.minimum_segment_size == tensorflow_tensorrt_config.minimum_segment_size
        assert tf_trt_model_configuration.trt_profiles == tensorflow_tensorrt_config.trt_profiles
        assert tf_trt_model_configuration.format == tensorflow_tensorrt_config.format
        assert tf_trt_model_configuration.parent_key == sm_model_configuration.key


def test_get_onnx_config_returns_model_configs_matching_custom_config_when_torch_framework_without_onnx_extended_conversion():  # noqa: E501
    onnx_config = OnnxConfig(
        opset=8, dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}}, graph_surgeon_optimization=False
    )
    model_configs = {Format.ONNX: []}
    custom_configs = [onnx_config]
    ModelConfigBuilder().get_onnx_config(Framework.TORCH, custom_configs, model_configs)

    assert len(model_configs[Format.ONNX]) == 1
    model_configuration = model_configs[Format.ONNX][0]
    assert isinstance(model_configuration, model_config.ONNXConfig)
    assert model_configuration.dynamic_axes == onnx_config.dynamic_axes
    assert model_configuration.opset == onnx_config.opset
    assert model_configuration.format == onnx_config.format
    assert model_configuration.parent_key is None
    assert model_configuration.graph_surgeon_optimization is onnx_config.graph_surgeon_optimization


def test_get_onnx_config_returns_model_configs_matching_custom_config_when_torch_framework_with_onnx_extended_conversion():  # noqa: E501
    torch_config = TorchConfig()
    onnx_config = OnnxConfig(
        opset=8,
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
        onnx_extended_conversion=True,
    )
    model_configs = {Format.TORCHSCRIPT: [], Format.ONNX: []}
    custom_configs = [torch_config, onnx_config]
    ModelConfigBuilder().get_torchscript_config(custom_configs, model_configs)
    ModelConfigBuilder().get_onnx_config(Framework.TORCH, custom_configs, model_configs)

    assert len(model_configs[Format.ONNX]) == 3
    for model_configuration, torchscript_model_configuration in zip(
        model_configs[Format.ONNX], [None] + model_configs[Format.TORCHSCRIPT]
    ):
        assert isinstance(model_configuration, model_config.ONNXConfig)
        assert model_configuration.dynamic_axes == onnx_config.dynamic_axes
        assert model_configuration.opset == onnx_config.opset
        assert model_configuration.format == onnx_config.format
        assert model_configuration.graph_surgeon_optimization is onnx_config.graph_surgeon_optimization
        if torchscript_model_configuration is None:
            assert model_configuration.parent_key is None
        else:
            assert isinstance(torchscript_model_configuration, model_config.TorchScriptConfig)
            assert model_configuration.parent_key == torchscript_model_configuration.key


def test_get_onnx_config_for_onnx_framework_returns_model_configs_matching_custom_config():
    onnx_config = OnnxConfig(opset=8, dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}})
    model_configs = {Format.ONNX: []}
    custom_configs = [onnx_config]
    ModelConfigBuilder().get_onnx_config(Framework.ONNX, custom_configs, model_configs)

    assert len(model_configs[Format.ONNX]) == 1
    model_configuration = model_configs[Format.ONNX][0]
    assert isinstance(model_configuration, model_config.ONNXConfig)
    assert model_configuration.dynamic_axes == onnx_config.dynamic_axes
    assert model_configuration.opset == onnx_config.opset
    assert model_configuration.format == onnx_config.format
    assert model_configuration.parent_key is None
    assert model_configuration.graph_surgeon_optimization is onnx_config.graph_surgeon_optimization


def test_get_onnx_config_for_jax_framework_returns_model_configs_matching_custom_config():
    savedmodel_config = TensorFlowConfig()
    onnx_config = OnnxConfig(opset=8, dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}})
    model_configs = {Format.TF_SAVEDMODEL: [], Format.ONNX: []}
    custom_configs = [savedmodel_config, onnx_config]
    ModelConfigBuilder().get_savedmodel_config(custom_configs, model_configs)
    ModelConfigBuilder().get_onnx_config(Framework.JAX, custom_configs, model_configs)

    assert len(model_configs[Format.ONNX]) == 1
    model_configuration = model_configs[Format.ONNX][0]
    for model_configuration, savedmodel_model_configuration in zip(
        model_configs[Format.ONNX], model_configs[Format.TF_SAVEDMODEL]
    ):
        assert isinstance(model_configuration, model_config.ONNXConfig)
        assert model_configuration.dynamic_axes == onnx_config.dynamic_axes
        assert model_configuration.opset == onnx_config.opset
        assert model_configuration.format == onnx_config.format
        assert model_configuration.graph_surgeon_optimization is onnx_config.graph_surgeon_optimization
        assert isinstance(savedmodel_model_configuration, model_config.TensorFlowSavedModelConfig)
        assert model_configuration.parent_key == savedmodel_model_configuration.key


def test_get_onnx_config_for_tensorflow_framework_returns_model_configs_matching_custom_config():
    savedmodel_config = TensorFlowConfig()
    onnx_config = OnnxConfig(opset=8, dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}})
    model_configs = {Format.TF_SAVEDMODEL: [], Format.ONNX: []}
    custom_configs = [savedmodel_config, onnx_config]
    ModelConfigBuilder().get_savedmodel_config(custom_configs, model_configs)
    ModelConfigBuilder().get_onnx_config(Framework.TENSORFLOW, custom_configs, model_configs)

    assert len(model_configs[Format.ONNX]) == 1
    for model_configuration, savedmodel_model_configuration in zip(
        model_configs[Format.ONNX], model_configs[Format.TF_SAVEDMODEL]
    ):
        assert isinstance(model_configuration, model_config.ONNXConfig)
        assert model_configuration.dynamic_axes == onnx_config.dynamic_axes
        assert model_configuration.opset == onnx_config.opset
        assert model_configuration.format == onnx_config.format
        assert model_configuration.graph_surgeon_optimization is onnx_config.graph_surgeon_optimization
        assert isinstance(savedmodel_model_configuration, model_config.TensorFlowSavedModelConfig)
        assert model_configuration.parent_key == savedmodel_model_configuration.key


def test_get_trt_config_returns_model_configs_matching_custom_config():
    onnx_config = OnnxConfig()
    trt_config = TensorRTConfig(
        precision=(TensorRTPrecision.FP16, TensorRTPrecision.FP16),
        precision_mode=TensorRTPrecisionMode.MIXED,
        max_workspace_size=10,
        trt_profiles=[TensorRTProfile().add("x", (1,), (2,), (3,))],
        optimization_level=1,
        compatibility_level=TensorRTCompatibilityLevel.AMPERE_PLUS,
    )

    custom_configs = [onnx_config, trt_config]
    model_configs = {Format.ONNX: [], Format.TENSORRT: []}
    ModelConfigBuilder().get_onnx_config(Framework.ONNX, custom_configs, model_configs)
    ModelConfigBuilder().get_trt_config(Framework.ONNX, custom_configs, model_configs)

    assert len(model_configs[Format.TENSORRT]) == 2
    for (onnx_model_configuration, precision), trt_model_configuration in zip(
        itertools.product(model_configs[Format.ONNX], trt_config.precision), model_configs[Format.TENSORRT]
    ):
        assert isinstance(trt_model_configuration, model_config.TensorRTConfig)
        assert isinstance(onnx_model_configuration, model_config.ONNXConfig)
        assert trt_model_configuration.precision == precision
        assert trt_model_configuration.max_workspace_size == trt_config.max_workspace_size
        assert trt_model_configuration.precision_mode == trt_config.precision_mode
        assert trt_model_configuration.format == trt_config.format
        assert trt_model_configuration.trt_profiles == trt_config.trt_profiles
        assert trt_model_configuration.parent_key == onnx_model_configuration.key
        assert trt_model_configuration.optimization_level == 1
        assert trt_model_configuration.compatibility_level == TensorRTCompatibilityLevel.AMPERE_PLUS
