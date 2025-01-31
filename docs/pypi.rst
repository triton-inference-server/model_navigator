..
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

Triton Model Navigator
========================

Welcome to the `Triton Model Navigator`_, an inference toolkit designed
for optimizing and deploying Deep Learning models with a focus on NVIDIA GPUs. The Triton Model Navigator streamlines the
process of moving models and pipelines implemented in `PyTorch`_, `TensorFlow`_, and `ONNX`_ to `TensorRT`_.

The Triton Model Navigator automates several critical steps, including model export, conversion, correctness testing, and
profiling. By providing a single entry point for various supported frameworks, users can efficiently search for the best
deployment option using the per-framework optimize function. The resulting optimized models are ready for deployment on
either `PyTriton`_ or `Triton Inference Server`_.

Features at Glance
--------------------

The distinct capabilities of the Triton Model Navigator are summarized in the feature matrix:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Feature
     - Description
   * - Ease-of-use
     - Single line of code to run all possible optimization paths directly from your source code
   * - Wide Framework Support
     - Compatible with various machine learning frameworks including PyTorch, TensorFlow, and ONNX
   * - Models Optimization
     - Enhance the performance of models such as ResNET and BERT for efficient inference deployment
   * - Pipelines Optimization
     - Streamline Python code pipelines for models such as Stable Diffusion and Whisper using Inplace Optimization, exclusive to PyTorch
   * - Model Export and Conversion
     - Automate the process of exporting and converting models between various formats with focus on TensorRT and Torch-TensorRT
   * - Correctness Testing
     - Ensures the converted model produce correct outputs validating against the original model
   * - Performance Profiling
     - Profiles models to select the optimal format based on performance metrics such as latency and throughput to optimize target hardware utilization
   * - Models Deployment
     - Automates models and pipelines deployment on PyTriton and Triton Inference Server through dedicated API

Documentation
---------------

Learn more about the Triton Model Navigator features in `documentation`_.

Prerequisites
---------------

Before proceeding with the installation of the Triton Model Navigator, ensure your system meets the following criteria:

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Version `3.9` or newer
- NVIDIA GPU

You can use NGC Containers for PyTorch and TensorFlow which contain all necessary dependencies:

- `PyTorch container`_
- `TensorFlow container`_

Install
---------

The Triton Model Navigator can be installed from `pypi.org` by running the following command:

.. code-block:: text

    pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[<extras,>]

Installing with PyTorch extras:

.. code-block:: text

    pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[torch]

Installing with TensorFlow extras:

.. code-block:: text

    pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[tensorflow]

Optimize Stable Diffusion with Inplace
----------------------------------------

The Inplace Optimize allows seamless optimization of models for deployment, such as converting
them to TensorRT, without requiring any changes to the original Python pipelines.

For the Stable Diffusion model, initialize the pipeline and wrap the model components with `nav.Module`:

.. code-block:: python

    import model_navigator as nav
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


    def get_pipeline():
        # Initialize Stable Diffusion pipeline and wrap modules for optimization
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.text_encoder = nav.Module(
            pipe.text_encoder,
            name="clip",
            output_mapping=lambda output: BaseModelOutputWithPooling(**output), # Mapping to convert output data to HuggingFace class
        )
        pipe.unet = nav.Module(
            pipe.unet,
            name="unet",
        )
        pipe.vae.decoder = nav.Module(
            pipe.vae.decoder,
            name="vae",
        )

        return pipe

Prepare a simple dataloader:

.. code-block:: python

    # Please mind, the first element in tuple need to be a batch size
    def get_dataloader():
        return [(1, "a photo of an astronaut riding a horse on mars")]

Execute model optimization:

.. code-block:: python

    pipe = get_pipeline()
    dataloader = get_dataloader()

    nav.optimize(pipe, dataloader)

Once the pipeline has been optimized, you can load explicit the most performant version of the modules executing:

.. code-block:: python

    nav.load_optimized()

After executing this method, when the optimized version of module exists, it will be used in your pipeline execution
directly in Python. The example how to serve Stable Diffusion pipeline through PyTriton can be
found `here`_.

Please read `Error isolation when running Python script`_ when you plan to place code in Python script.

Optimize ResNET and deploy on Triton
--------------------------------------

The Triton Model Navigator also supports an optimization path for deployment on Triton. This path is supported for nn.Module,
keras.Model or ONNX files which inputs are tensors.


To optimize ResNet50 model from TorchHub run the following code:

.. code-block:: python

    import torch
    import model_navigator as nav

    # Optimize Torch model loaded from TorchHub
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).eval()

    # Wrap model in nav.Module
    resnet50 = nav.Module(resnet50, name="resnet50")

    # Optimize Torch model loaded from TorchHub
    nav.optimize(resnet50, dataloader=[(1, [torch.randn(1, 3, 256, 256)])])


Please read `Error isolation when running Python script`_ when you plan to place code in Python script.

Once optimization is done, creating a model store for deployment on Triton is simple as following code:

.. code-block:: python

    import pathlib

    # Generate the model store from optimized model
    resnet50.triton_model_store(
        model_repository_path=pathlib.Path("model_repository"),
    )

Profile any model or callable in Python
-----------------------------------------

The Triton Model Navigator enhances models and pipelines and provides a uniform method for profiling any Python
function, callable, or model. At present, our support is limited strictly to static batch profiling scenarios.

As an example, we will use a simple function that simply sleeps for 50 ms:

.. code-block:: python

    import time


    def custom_fn(input_):
        # wait 50ms
        time.sleep(0.05)
        return input_

Let's provide a dataloader we will use for profiling:

.. code-block:: python

    # Tuple of batch size and data sample
    dataloader = [(1, ["This is example input"])]

Finally, run the profiling of the function with prepared dataloader:

.. code-block:: python

    nav.profile(custom_fn, dataloader)


Error isolation when running Python script
------------------------------------------

**Important**: Please review below section to prevent unexpected issues when running `optimize`.

For better error isolation, some conversions and exports are run in separate child processes using multiprocessing in
the `spawn` mode. This means that everything in a global scope will be run in a child process. You can encounter
unexpected issue when the optimization code is place in Python script and executed as:

.. code-block:: shell

    python optimize.py

To prevent nested optimization, you have to either put the optimize code in:

.. code-block:: python

    if __name__ == "__main__":
        # optimization goes here

or

.. code-block:: python

    import multiprocessing as mp
    if mp.current_process().name == "MainProcess":
        # optimization goes here


If none of the above works for you, you can run all optimization in a single process at the cost of error isolation by
setting the following environment variable:

.. code-block:: shell

    NAVIGATOR_USE_MULTIPROCESSING=False


Examples
----------

We offer comprehensive, step-by-step `guides`_ that showcase the utilization of the Triton Model Navigator’s diverse
features. These
guides are designed to elucidate the processes of optimization, profiling, testing, and deployment of models using
`PyTriton`_ and `Triton Inference Server`_.


Links
-------

* Documentation: https://triton-inference-server.github.io/model_navigator
* Source: https://github.com/triton-inference-server/model_navigator
* Issues: https://github.com/triton-inference-server/model_navigator/issues
* Examples: https://github.com/triton-inference-server/model_navigator/tree/main/examples.
* Changelog: https://github.com/triton-inference-server/model_navigator/blob/main/CHANGELOG.md
* Known Issues: https://github.com/triton-inference-server/model_navigator/blob/main/docs/known_issues.md
* Contributing: https://github.com/triton-inference-server/model_navigator/blob/main/CONTRIBUTING.md

.. _Triton Model Navigator: https://github.com/triton-inference-server/model_navigator
.. _Triton Inference Server: https://github.com/triton-inference-server/server
.. _TensorRT: https://github.com/NVIDIA/TensorRT
.. _PyTriton: https://github.com/triton-inference-server/pytriton
.. _documentation: https://triton-inference-server.github.io/model_navigator
.. _PyTorch container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
.. _TensorFlow container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow
.. _here: https://github.com/triton-inference-server/pytriton/tree/main/examples/huggingface_stable_diffusion
.. _PyTorch: https://pytorch.org
.. _TensorFlow: https://www.tensorflow.org
.. _ONNX: https://onnx.ai
.. _guides: https://github.com/triton-inference-server/model_navigator/tree/main/examples
