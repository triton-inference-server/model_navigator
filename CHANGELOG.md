<!--
Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Changelog

## 0.10.0 (unreleased)
- new: inplace `nav.Module` accepts `batching` flag which overrides a config setting and `precision` which allows setting appropriate configuration for TensorRT
- new: Allow to set device when loading optimized modules using `nav.load_optimized()`
- fix: Maintaining modules device in `nav.profile()`


## 0.9.0

- new: TensorRT Timing Tactics Cache Management - using timing tactics cache files for optimization performance improvements
- new: Added throughput saturation verification in `nav.profile()` (enabled by default)
- new: Allow to override Inplace cache dir through `MODEL_NAVIGATOR_DEFAULT_CACHE_DIR` env variable
- new: inplace `nav.Module` can now receive a function name to be used instead of __call__ in modules/submodules, allows customizing modules with non-standard calls
- fix: torch dynamo export and torch dynamo onnx export
- fix: measurement stabilization in `nav.profile()`
- fix: inplace inference through Torch
- fix: trt_profiles argument handling in ONNX to TRT conversion
- fix: optimal shape configuration for batch size in Inplace API
- change: Disable TensorRT profile builder
- change: `nav.optimize()` does not override module configuration

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.3.0a0+6ddf5cf85e](https://github.com/pytorch/pytorch/commit/40ec155e58ee1a1921377ff921b55e61502e4fb3)
    - [TensorFlow 2.15.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0)
    - [TensorRT 8.6.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [Torch-TensorRT 2.0.0.dev0](https://github.com/NVIDIA/Torch-TensorRT)
    - [ONNX Runtime 1.17.1](https://github.com/microsoft/onnxruntime/tree/v1.17.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.4
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.16.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.16.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.8.1

- fix: Inference with TensorRT when model has input with empty shape
- fix: Using stabilized runners when model has no batching
- fix: Invalid dependencies for cuDNN - review [known issues](docs/known_issues.md)
- fix: Make ONNX Graph Surgeon produce artifacts within protobuf Limit (2G)
- change: Remove TensorRTCUDAGraph from default runners
- change: updated ONNX package to 1.16

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.3.0a0+40ec155e58](https://github.com/pytorch/pytorch/commit/40ec155e58)
    - [TensorFlow 2.15.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0)
    - [TensorRT 8.6.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [Torch-TensorRT 2.0.0.dev0](https://github.com/NVIDIA/Torch-TensorRT)
    - [ONNX Runtime 1.17.1](https://github.com/microsoft/onnxruntime/tree/v1.17.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.4
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.16.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.16.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.8.0

- new: Allow to select device for TensorRT runner
- new: Add device output buffers to TensorRT runner
- new: nav.profile added for profiling any Python function
- change: API for Inplace optimization (breaking change)
- fix: Passing inputs for Torch to ONNX export
- fix: Parse args to kwargs in torchscript-trace export
- fix: Lower peak memory usage when loading Torch inplace optimized model

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.3.0a0+ebedce2](https://github.com/pytorch/pytorch/commit/ebedce2)
    - [TensorFlow 2.15.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0)
    - [TensorRT 8.6.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [Torch-TensorRT 2.0.0.dev0](https://github.com/NVIDIA/Torch-TensorRT)
    - [ONNX Runtime 1.17.1](https://github.com/microsoft/onnxruntime/tree/v1.17.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.4
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.16.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.16.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.7

- change: Add input and output specs for Triton model repositories generated from packages

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.2.0a0+81ea7a48](https://github.com/pytorch/pytorch/commit/81ea7a48)
    - [TensorFlow 2.14.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.14.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.16.2](https://github.com/microsoft/onnxruntime/tree/v1.16.2)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.16.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.16.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.6

- fix: Passing inputs for Torch to ONNX export
- fix: Passing input data to OnnxCUDA runner

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.2.0a0+81ea7a48](https://github.com/pytorch/pytorch/commit/81ea7a48)
    - [TensorFlow 2.14.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.14.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.16.2](https://github.com/microsoft/onnxruntime/tree/v1.16.2)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.16.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.16.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.5

- new: FP8 precision support for TensorRT
- new: Support for autocast and inference mode configuration for Torch runners
- new: Allow to select device for Torch and ONNX runners
- new: Add support for `default_model_filename` in Triton model configuration
- new: Detailed profiling of inference steps (pre- and postprocessing, memcpy and compute)
- fix: JAX export and TensorRT conversion fails when custom workspace is used
- fix: Missing max workspace size passed to TensorRT conversion
- fix: Execution of TensorRT optimize raise error during handling output metadata
- fix: Limited Polygraphy version to work correctly with onnxruntime-gpu package

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.2.0a0+6a974be](https://github.com/pytorch/pytorch/commit/6a974be)
    - [TensorFlow 2.13.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.13.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.16.2](https://github.com/microsoft/onnxruntime/tree/v1.16.2)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.15.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.15.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.4

- new: decoupled mode configuration in Triton Model Config
- new: support for PyTorch ExportedProgram and ONNX dynamo export
- new: added GraphSurgeon ONNX optimization
- fix: compatibility of generating PyTriton model config through adapter
- fix: installation of packages that are platform dependent
- fix: update package config with model loaded from source
- change: in TensorRT runner, when TensorType.TORCH is the return type lazily convert tensor to Torch
- change: move from Polygraphy CLI to Polygraphy Python API
- change: removed Windows from support list

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+32f93b1](https://github.com/pytorch/pytorch/commit/32f93b1)
    - [TensorFlow 2.13.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.13.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.16.2](https://github.com/microsoft/onnxruntime/tree/v1.16.2)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.49.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.15.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.15.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.3

- new: Data dependent dynamic control flow support in nav.Module (multiple computation graphs per module)
- new: Added find max batch size utility
- new: Added utilities API documentation
- new: Add Timer class for measuring execution time of models and Inplace modules.
- fix: Use wide range of shapes for TensorRT conversion
- fix: Sorting of samples loaded from workspace
- change: in Inplace, store one sample by default per module and store shape info for all samples
- change: always execute export for all supported formats


- Known issues and limitations:
    - nav.Module moves original torch.nn.Module to the CPU, in case of weight sharing that might result in unexpected
      behaviour
    - For data dependent dynamic control flow (multiple computation graphs) nav.Module might copy the weights for each
      separate graph

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+29c30b1](https://github.com/pytorch/pytorch/commit/29c30b1)
    - [TensorFlow 2.13.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.13.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.15.1](https://github.com/microsoft/onnxruntime/tree/v1.15.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.15.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.15.1)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.2

- fix: Obtaining inputs names from ONNX file for TensorRT conversion
- change: Raise exception instead of exit with code when required command has failed

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+b5021ba](https://github.com/pytorch/pytorch/commit/b5021ba9)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.15.1](https://github.com/microsoft/onnxruntime/tree/v1.15.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.1

- fix: gather onnx input names based on model's forward signature
- fix: do not run TensorRT max batch size search when max batch size is None
- fix: use pytree metadata to flatten torch complex outputs

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+b5021ba](https://github.com/pytorch/pytorch/commit/b5021ba9)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.15.1](https://github.com/microsoft/onnxruntime/tree/v1.15.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.7.0

- new: Inplace Optimize feature - optimize models directly in the Python code
- new: Non-tensor inputs and outputs support
- new: Model warmup support in Triton model configuration
- new: nav.tensorrt.optimize api added for testing and measuring performance of TensorRT models
- new: Extended custom configs to pass arguments directly to export and conversion operations like `torch.onnx.export`
  or `polygraphy convert`
- new: Collect GPU clock during model profiling
- new: Add option to configure minimal trials and stabilization windows for performance verification and profiling
- change: Navigator package version change to 0.2.3. Custom configurations now use trt_profiles list instead single
  value
- change: Store separate reproduction scripts for runners used during correctness and profiling

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+b5021ba](https://github.com/pytorch/pytorch/commit/b5021ba9)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.15.1](https://github.com/microsoft/onnxruntime/tree/v1.15.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.27
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.6.3

- fix: Conditional imports of supported frameworks in export commands

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+4136153](https://github.com/pytorch/pytorch/commit/4136153)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.6.2

- new: Collect information about TensorRT shapes used during conversion
- fix: Invalid link in documentation
- change: Improved rendering documentation

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+4136153](https://github.com/pytorch/pytorch/commit/4136153)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.6.1

- fix: Add model from package to Triton model store with custom configs

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+4136153](https://github.com/pytorch/pytorch/commit/4136153)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.6.0

- new: Zero-copy runners for Torch, ONNX and TensorRT - omit H2D and D2H memory copy between runners execution
- new: `nav.pacakge.profile` API method to profile generated models on provided dataloader
- change: ProfilerConfig replaced with OptimizationProfile:
    - new: OptimizationProfile impact the conversion for TensorRT
    - new: `batch_sizes` and `max_batch_size` limit the max profile in TensorRT conversion
    - new: Allow to provide separate dataloader for profiling - first sample used only
- new: allow to run `nav.package.optimize` on empty package - status generation only
- new: use `torch.inference_mode` for inference runner when PyTorch 2.x is available
- fix: Missing `model` in config when passing package generated during `nav.{framework}.optimize` directly
  to `nav.package.optimize` command
- Other minor fixes and improvements

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+4136153](https://github.com/pytorch/pytorch/commit/4136153)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.6

- fix: Load samples as sorted to keep valid order
- fix: Execute conversion when model already exists in path
- Other minor fixes and improvements

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+fe05266f](https://github.com/pytorch/pytorch/commit/fe05266f)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.5

- new: Public `nav.utilities` module with UnpackedDataloader wrapper
- new: Added support for strict flag in Torch custom config
- new: Extended TensorRT custom config to support builder optimization level and hardware compatibility flags
- fix: Invalid optimal shape calculation for odd values in max batch size

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+fe05266f](https://github.com/pytorch/pytorch/commit/fe05266f)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.4

- new: Custom implementation for ONNX and TensorRT runners
- new: Use CUDA 12 for JAX in unit tests and functional tests
- new: Step-by-step examples
- new: Updated documentation
- new: TensorRTCUDAGraph runner introduced with support for CUDA graphs
- fix: Optimal shape not set correctly during adaptive conversion
- fix: Find max batch size command for JAX
- fix: Save stdout to logfiles in debug mode

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.1.0a0+fe05266f](https://github.com/pytorch/pytorch/commit/fe05266f)
    - [TensorFlow 2.12.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0)
    - [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.47.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.3

- fix: filter outputs using output_metadata in ONNX runners

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.0.0a0+1767026](https://github.com/pytorch/pytorch/commit/1767026)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.2

- new: Added Contributor License Agreement (CLA)
- fix: Added missing --extra-index-url to installation instruction for pypi
- fix: Updated wheel readme
- fix: Do not run TorchScript export when only ONNX in target formats and ONNX extended export is disabled
- fix: Log full traceback for ModelNavigatorUserInputError

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 2.0.0a0+1767026](https://github.com/pytorch/pytorch/commit/1767026)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3.1](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.26
    - [tf2onnx v1.14.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.14.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.1

- fix: Using relative workspace cause error during Onnx to TensorRT conversion
- fix: Added external weight in package for ONNX format
- fix: bugfixes for functional tests

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.5.0

- new: Support for PyTriton deployment
- new: Support for Python models with python.optimize API
- new: PyTorch 2 compile CPU and CUDA runners
- new: Collect conversion max batch size in status
- new: PyTorch runners with `compile` support
- change: Improved handling CUDA and CPU runners
- change: Reduced finding device max batch size time by running it once as separate pipeline
- change: Stored find max batch size result in separate filed in status

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.4.4

- fix: when exporting single input model to saved model, unwrap one element list with inputs

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.4.3

- fix: in Keras inference use model.predict(tensor) for single input models

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.4.2

- fix: loading configuration for trt_profile from package
- fix: missing reproduction scripts and logs inside package
- fix: invalid model path in reproduction script for ONNX to TRT conversion
- fix: collecting metadata from ONNX model in main thread during ONNX to TRT conversion

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.3](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.44.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.4.1

- fix: when specified use dynamic axes from custom OnnxConfig

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.2.2](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.43.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.4.0

- new: `optimize` method that replace `export` and perform max batch size search and improved profiling during process
- new: Introduced custom configs in `optimize` for better parametrization of export/conversion commands
- new: Support for adding user runners for model correctness and profiling
- new: Search for max possible batch size per format during conversion and profiling
- new: API for creating Triton model store from Navigator Package and user provided models
- change: Improved status structure for Navigator Package
- deprecated: Optimize for Triton Inference Server support
- deprecated: HuggingFace contrib module
- Bug fixes and other improvements

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [PyTorch 1.14.0a0+410ce96](https://github.com/pytorch/pytorch/commit/410ce96)
    - [TensorFlow 2.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
    - [TensorRT 8.5.2.2](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
    - [ONNX Runtime 1.13.1](https://github.com/microsoft/onnxruntime/tree/v1.13.1)
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.43.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.4.6
    - [tf2onnx v1.13.0](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.13.0)
    - Other component versions depend on the used framework containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.8

- Updated NVIDIA containers defaults to 22.11

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.42.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.20.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.12.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.12.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.7

- Updated NVIDIA containers defaults to 22.10

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.42.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.20.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.12.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.12.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.6

- Updated NVIDIA containers defaults to 22.09
- Model Navigator Export API:
    - new: cast int64 input data to int32 in runner for Torch-TensorRT
    - new: cast 64-bit data samples to 32-bit values for TensorRT
    - new: verbose flag for logging export and conversion commands to console
    - new: debug flag to enable debug mode for export and conversion commands
    - change: logs from commands are streamed to console during command run
    - change: package load omit the log files and autogenerated scripts

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.42.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.20.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.12.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.12.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.5

- Updated NVIDIA containers defaults to 22.08
- Model Navigator Export API:
    - new: TRTExec runner use `use_cuda_graph=True` by default
    - new: log warning instead of raising error when dataloader dump inputs with `nan` or `inf` values
    - new: enabled logging for command input parameters
    - fix: invalid use of Polygraphy TRT profile when trt_dynamic_axes is passed to export function

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.38.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.19.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.12.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.12.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.4

- Updated NVIDIA containers defaults to 22.07
- Model Navigator OTIS:
    - deprecated: `TF32` precision for TensorRT from CLI options - will be removed in future versions
    - fix: Tensorflow module was imported when obtaining model signature during conversion
- Model Navigator Export API:
    - new: Support for building framework containers with Model Navigator installed
    - new: Example for loading Navigator Package for reproducing the results
    - new: Create reproducing script for correctness and performance steps
    - new: TrtexecRunner for correctness and performance tests
      with [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) tool
    - new: Use TF32 support by default for models with FP32 precision
    - new: Reset conversion parameters to defaults when using `load` for package
    - new: Testing all options for JAX export enable_xla and jit_compile parameters
    - change: Profiling stability improvements
    - change: Rename of `onnx_runtimes` export function parameters to `runtimes`
    - deprecated: `TF32` precision for TensorRT from available options - will be removed in future versions
    - fix: Do not save TF-TRT models to the .nav package
    - fix: Do not save TF-TRT models from the .nav package
    - fix: Correctly load .nav packages when `_input_names` or `_output_names` specified
    - fix: Adjust TF and TF-TRT model signatures to match `input_names`
    - fix: Save ONNX opset for CLI configuration inside package
    - fix: Reproduction scripts were missing for failing paths

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.38.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.17.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.11.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.11.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.3

- Model Navigator Export API:
    - new: Improved handling inputs and outputs metadata
    - new: Navigator Package version updated to 0.1.3
    - new: Backward compatibility with previous versions of Navigator Package
    - fix: Dynamic shapes for output shapes were read incorrectly

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.36.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.17.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.11.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.11.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.2

- Updated NVIDIA containers defaults to 22.06
- Model Navigator OTIS:
    - new: Perf Analyzer profiling data use base64 format for content
    - fix: Signature for TensorRT model when has `uint64` or `int64` input and/or outputs defined
- Model Navigator Export API:
    - new: Updated navigator package format to 0.1.1
    - new: Added Model Navigator version to status file
    - new: Add atol and rtol configuration to CLI config for model
    - new: Added experimental support for JAX models
    - new: In case of export or conversion failures prepare minimal scripts to reproduce errors
    - fix: Conversion parameters are not stored in Navigator Package for CLI execution

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.36.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.17.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.11.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.11.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.1

- Updated NVIDIA containers defaults to 22.05
- Model Navigator OTIS:
    - fix: Saving paths inside the Triton package status file
    - fix: Empty list of gpus cause the process run on CPU only
    - fix: Reading content from zipped Navigator Package
    - fix: When no GPU or target device set to CPU `optimize` avoid running unsupported conversions in CLI
    - new: Converter accept passing target device kind to selected CPU or GPU supported conversions
    - new: Added support for OpenVINO accelerator for ONNXRuntime
    - new: Added option `--config-search-early-exit-enable` for Model Analyzer early exit support
      in manual profiling mode
    - new: Added option `--model-config-name` to the `select` command.
      It allows to pick a particular model configuration for deployment from the set of all configurations
      generated by Triton Model Analyzer, even if it's not the best performing one.
    - removed: The `--tensorrt-strict-types` option has been removed due to deprecation of the functionality
      in upstream libraries.
- Model Navigator Export API:
    - new: Added dynamic shapes support and trt dynamic shapes support for TensorFlow2 export
    - new: Improved per format logging
    - new: PyTorch to Torch-TRT precision selection added
    - new: Advanced profiling (measurement windows, configurable batch sizes)

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.36.2
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.19
    - [Triton Model Analyzer 1.16.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.10.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.10.1)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.3.0

- Updated NVIDIA containers defaults to 22.04
- Model Navigator Export API
    - Support for exporting models from TensorFlow2 and PyTorch source code to supported target formats
    - Support for conversion from ONNX to supported target formats
    - Support for exporting HuggingFace models
    - Conversion, Correctness and performance tests for exported models
    - Definition of package structure for storing all exported models and additional metadata
- Model Navigator OTIS:
    - change: `run` command has been deprecated and may be removed in a future release
    - new: `optimize` command replace `run` and produces an output `*.triton.nav` package
    - new: `select` selects the best-performing configuration from `*.triton.nav` package and create a
      Triton Inference Server model repository
    - new: Added support for using shared memory option for Perf Analyzer
- Remove wkhtmltopdf package dependency

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.35.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.14
    - [Triton Model Analyzer 1.14.0](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.9.3](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.3)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.2.7

- Updated NVIDIA containers defaults to 22.02
- Removed support for Python 3.7
- Triton Model configuration related:
    - Support dynamic batching without setting preferred batch size value
- Profiling related:
    - Deprecated `--config-search-max-preferred-batch-size` flag as is no longer supported in Triton Model Analyzer

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.35.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.14
    - [Triton Model Analyzer 1.8.2](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.9.3](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.3)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

## 0.2.6

- Updated NVIDIA containers defaults to 22.01
- Removed support for Python 3.6 due to EOL
- Conversion related:
    - Added support for Torch-TensorRT conversion
- Fixes and improvements
    - Processes inside containers started by Model Navigator now run without root privileges
    - Fix for volume mounts while running Triton Inference Server in container from other container
    - Fix for conversion of models without file extension on input and output paths
    - Fix using `--model-format` argument when input and output files have no extension

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.35.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.14
    - [Triton Model Analyzer 1.8.2](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.9.3](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.3)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - no verification of conversion results for conversions: TF -> ONNX, TF->TF-TRT, TorchScript -> ONNX
    - possible to define a single profile for TensorRT
    - no custom ops support
    - Triton Inference Server stays in the background when the profile
      process is interrupted by the user
    - TF-TRT conversion lost outputs shapes info

## 0.2.5

- Updated NVIDIA containers defaults to 21.12
- Conversion related:
    - [Experimental] TF-TRT - fixed default dataset profile generation
- Configuration Model on Triton related
    - Fixed name for onnxruntime backend in Triton model deployment configuration

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.33.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.14
    - [Triton Model Analyzer 1.8.2](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.9.3](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.3)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - no verification of conversion results for conversions: TF -> ONNX, TF->TF-TRT, TorchScript -> ONNX
    - possible to define a single profile for TensorRT
    - no custom ops support
    - Triton Inference Server stays in the background when the profile
      process is interrupted by the user
    - TF-TRT conversion lost outputs shapes info

## 0.2.4

- Updated NVIDIA containers defaults to 21.10
- Fixed generating profiling data when `dtypes` are not passed
- Conversion related:
    - [Experimental] Added support for TF-TRT conversion
- Configuration Model on Triton related
    - Added possibility to select batching mode - default, dynamic and disabled options supported
- Install dependencies from pip packages instead of wheels for Polygraphy and Triton Model Analyzer
- fixes and improvements

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.33.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.14
    - [Triton Model Analyzer 1.8.2](https://github.com/triton-inference-server/model_analyzer)
    - tf2onnx: [v1.9.3](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.3)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - no verification of conversion results for conversions: TF -> ONNX, TF->TF-TRT, TorchScript -> ONNX
    - possible to define a single profile for TensorRT
    - no custom ops support
    - Triton Inference Server stays in the background when the profile
      process is interrupted by the user
    - TF-TRT conversion lost outputs shapes info

## 0.2.3

- Updated NVIDIA containers defaults to 21.09
- Improved naming of arguments specific for TensorRT conversion and acceleration with backward compatibility
- Use pip package for Triton Model Analyzer installation with minimal version 1.8.0
- Fixed `model_repository` path to be not relative to `<navigator_workspace>` dir
- Handle exit codes correctly from CLI commands
- Support for use device ids for `--gpus` argument
- Conversion related
    - Added support for precision modes to support multiple precisions during conversion to TensorRT
    - Added `--tensorrt-sparse-weights` flag for sparse weight optimization for TensorRT
    - Added `--tensorrt-strict-types` flag forcing it to choose tactics based on the layer precision for TensorRT
    - Added `--tensorrt-explicit-precision` flag enabling explicit precision mode
    - Fixed nan values appearing in relative tolerance during conversion to TensorRT
- Configuration Model on Triton related
    - Removed default value for `engine_count_per_device`
    - Added possibility to define Triton Custom Backend parameters with `triton_backend_parameters` command
    - Added possibility to define max workspace size for TensorRT backend accelerator using
      argument `tensorrt_max_workspace_size`
- Profiling related
    - Added `config_search` prefix to all profiling parameters (BREAKING CHANGE)
    - Added `config_search_max_preferred_batch_size` parameter
    - Added `config_search_backend_parameters` parameter
- fixes and improvements

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Versions of used external components:
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.32.0
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.13
    - tf2onnx: [v1.9.2](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.2) (support for ONNX opset 14,
      tf 1.15 and 2.6)
    - [Triton Model Analyzer 1.8.2](https://github.com/triton-inference-server/model_analyzer)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - possible to define a single profile for TensorRT

## 0.2.2

- Updated NVIDIA containers defaults to 21.08

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 1.7.0
    - [Triton Inference Server Client](https://github.com/triton-inference-server/client/): 2.13.0
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.31.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.11
    - tf2onnx: [v1.9.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.1) (support for ONNX opset 14,
      tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - possible to define a single profile for TensorRT

## 0.2.1

- Fixed triton-model-config error when tensorrt_capture_cuda_graph flag is not passed
- Dump Conversion Comparator inputs and outputs into JSON files
- Added information in logs on the tolerance parameters values to pass the conversion verification
- Use `count_windows` mode as default option for Perf Analyzer
- Added possibility to define custom docker images
- Bugfixes

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 1.6.0
    - [Triton Inference Server Client](https://github.com/triton-inference-server/client/): 2.12.0
    - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/): 0.31.1
    - [GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/): 0.3.11
    - tf2onnx: [v1.9.1](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.9.1) (support for ONNX opset 14,
      tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - possible to define a single profile for TensorRT
    - TensorRT backend acceleration not supported for ONNX Runtime in Triton Inference Server ver. 21.07

## 0.2.0

- comprehensive refactor of command-line API in order to provide more gradual
  pipeline steps execution

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 21.05
    - tf2onnx: [v1.8.5](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.8.5) (support for ONNX opset 13,
      tf 1.15 and 2.5)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      See its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues and limitations
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - issues with TorchScript -> ONNX conversion due
      to [issue in PyTorch 1.8](https://github.com/pytorch/pytorch/issues/53506)
        - affected NVIDIA PyTorch containers: 20.12, 21.02, 21.03
        - workaround: use PyTorch containers newer than 21.03
    - possible to define a single profile for TensorRT

## 0.1.1

- documentation update

## 0.1.0

- Release of main components:
    - Model Converter - converts the model to a set of variants optimized for inference or to be later optimized by
      Triton Inference Server backend.
    - Model Repo Builder - setup Triton Inference Server Model Repository, including its configuration.
    - Model Analyzer - select optimal Triton Inference Server configuration based on models compute and memory
      requirements,
      available computation infrastructure, and model application constraints.
    - Helm Chart Generator - deploy Triton Inference Server and model with optimal configuration to cloud.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Versions of used external components:
    - [Triton Model Analyzer](https://github.com/triton-inference-server/model_analyzer): 21.03+616e8a30
    - tf2onnx: [v1.8.4](https://github.com/onnx/tensorflow-onnx/releases/tag/v1.8.4) (support for ONNX opset 13, tf 1.15
      and 2.4)
    - Other component versions depend on the used framework and Triton Inference Server containers versions.
      Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
      for a detailed summary.

[//]: <> (keep up to date list of known issues inside docs/known_issue.md and paste it here on major and minor release)

- Known issues
    - missing support for stateful models (ex. time-series one)
    - missing support for models without batching support
    - no verification of conversion results for conversions: TF -> ONNX, TorchScript -> ONNX
    - issues with TorchScript -> ONNX conversion due
      to [issue in PyTorch 1.8](https://github.com/pytorch/pytorch/issues/53506)
        - affected NVIDIA PyTorch containers: 20.12, 21.03
        - workaround: use containers different from above
    - Triton Inference Server stays in the background when the profile process is interrupted by the user
