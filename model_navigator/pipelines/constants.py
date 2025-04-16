# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Constants definition of pipelines."""

# Pipeline names
PIPELINE_CORRECTNESS = "Correctness"
PIPELINE_FIND_MAX_BATCH_SIZE = "Finding max batch size for fixed shapes based pipelines"
PIPELINE_PERFORMANCE = "Performance"
PIPELINE_PREPROCESSING = "Preprocessing"
PIPELINE_PROFILING = "Profiling"
PIPELINE_TF_TENSORRT = "TensorFlow-TensorRT Conversion"
PIPELINE_TF2_CONVERSION = "TensorFlow 2 Conversion"
PIPELINE_TF2_EXPORT = "TensorFlow2 Export"
PIPELINE_TENSORRT_CONVERSION = "TensorRT Conversion"
PIPELINE_TORCH_TENSORRT_CONVERSION = "Torch-TensorRT Conversion"
PIPELINE_TORCH_CONVERSION = "PyTorch Conversion"
PIPELINE_TORCH_EXPORT = "PyTorch Export"
PIPELINE_TORCH_EXPORTEDPROGRAM = "PyTorch ExportedProgram Export"
PIPELINE_VERIFY_MODELS = "Verify Models"
