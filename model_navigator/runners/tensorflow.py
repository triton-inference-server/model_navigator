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
"""Runner definition for TensorFlow based models."""

import gc
from collections import OrderedDict
from typing import Dict, List, Mapping

import numpy as np

from model_navigator.api.config import Format
from model_navigator.core.dataloader import get_default_output_names
from model_navigator.runners.base import DeviceKind, NavigatorRunner
from model_navigator.runners.registry import register_runner
from model_navigator.utils import module

tf = module.lazy_import("tensorflow")


class _BaseTFRunner(NavigatorRunner):
    """Runs inference using TensorFlow2."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """Runner initialization implementation."""
        self._loaded_model = None

    def deactivate_impl(self):
        """Runner deactivation implementation."""
        self._loaded_model = None

    def infer_impl(self, feed_dict: Dict, *args, **kwargs):
        """Runner inference implementation override."""
        outputs = self._infer_impl(feed_dict)

        if self.output_metadata is None:
            return outputs

        if self.output_metadata:
            output_names = self.output_metadata.keys()
        else:
            output_names = outputs.keys() if isinstance(outputs, Mapping) else get_default_output_names(len(outputs))

        if isinstance(outputs, np.ndarray):
            outputs = (outputs,)
        if isinstance(outputs, Mapping):
            outputs = outputs.values()

        out_dict = OrderedDict()
        for name, output in zip(output_names, outputs):
            out_dict[name] = output

        return out_dict

    def _infer_impl(self, feed_dict: Dict):
        raise NotImplementedError


class TensorFlowSavedModelCUDARunner(_BaseTFRunner):
    """Runs inference for TensorFlow SavedModels."""

    def activate_impl(self):
        """Runner activation implementation."""
        self._loaded_model = tf.keras.models.load_model(str(self._model))

    def _infer_impl(self, feed_dict: Dict):
        """Runner inference handler implementation."""
        infer = self._loaded_model.signatures["serving_default"]
        outputs = [output.numpy() for output in infer(**feed_dict).values()]
        return outputs

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TF_SAVEDMODEL

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorFlowSavedModelCUDA"


class TensorFlowSavedModelCPURunner(_BaseTFRunner):
    """Runs inference for TensorFlow SavedModels."""

    def activate_impl(self):
        """Runner activation implementation."""
        if any(device.device_type == "GPU" for device in tf.config.get_visible_devices()):
            tf.config.set_visible_devices([], "GPU")
        self._loaded_model = tf.keras.models.load_model(str(self._model))

    def deactivate_impl(self):
        """Runner deactivation implementation."""

    def _infer_impl(self, feed_dict: Dict):
        """Runner inference handler implementation."""
        infer = self._loaded_model.signatures["serving_default"]
        outputs = [output.numpy() for output in infer(**feed_dict).values()]
        return outputs

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TF_SAVEDMODEL

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorFlowSavedModelCPU"


class TensorFlowTensorRTRunner(TensorFlowSavedModelCUDARunner):
    """Runs inference for TensorFlow TensorRT models."""

    @classmethod
    def format(cls) -> Format:
        """Runner supported format."""
        return Format.TF_TRT

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorFlowTensorRT"

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]


class TensorFlowCUDARunner(_BaseTFRunner):
    """Runs inference for TensorFlow models in source."""

    def activate_impl(self):
        """Runner activation implementation."""
        self._loaded_model = self.model

    def deactivate_impl(self):
        """Runner deactivation implementation."""
        super().deactivate_impl()
        # TODO: this does not allow other processes to use the memory, but allows TF to use it
        tf.keras.backend.clear_session()
        gc.collect()

    def _infer_impl(self, feed_dict):
        """Runner inference handler implementation."""
        inputs = self.input_metadata.unflatten_sample(feed_dict)
        outputs = self.model.predict(inputs, verbose=0)
        return outputs

    @classmethod
    def format(cls) -> Format:
        """Format for runner."""
        return Format.TENSORFLOW

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CUDA]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorFlowCUDA"


class TensorFlowCPURunner(_BaseTFRunner):
    """Runs inference for TensorFlow models in source."""

    def activate_impl(self):
        """Runner activation implementation."""
        if any(device.device_type == "GPU" for device in tf.config.get_visible_devices()):
            tf.config.set_visible_devices([], "GPU")
        self._loaded_model = self.model

    def deactivate_impl(self):
        """Runner deactivation implementation."""
        super().deactivate_impl()
        # TODO: this does not allow other processes to use the memory, but allows TF to use it
        tf.keras.backend.clear_session()
        gc.collect()

    def _infer_impl(self, feed_dict):
        """Runner inference handler implementation."""
        inputs = self.input_metadata.unflatten_sample(feed_dict)
        outputs = self.model.predict(inputs, verbose=0)
        return outputs

    @classmethod
    def format(cls) -> Format:
        """Format for runner."""
        return Format.TENSORFLOW

    @classmethod
    def devices_kind(cls) -> List[DeviceKind]:
        """Return supported devices for runner."""
        return [DeviceKind.CPU]

    @classmethod
    def name(cls) -> str:
        """Runner name."""
        return "TensorFlowCPU"


def register_tensorflow_runners():
    """Register TensorFlow runner in global registry."""
    register_runner(TensorFlowSavedModelCPURunner)
    register_runner(TensorFlowSavedModelCUDARunner)
    register_runner(TensorFlowTensorRTRunner)
    register_runner(TensorFlowCUDARunner)
    register_runner(TensorFlowCPURunner)
