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

import time
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy
import tensorflow as tf  # pytype: disable=import-error
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata

from model_navigator.framework_api.utils import Framework, validate_sample_output


class TFRunner(BaseRunner):
    """
    Runs inference using TensorFlow2.
    """

    def __init__(self, model, input_metadata, output_names, name=None):
        """
        Args:
            model (tensorflow.keras.Model) or (str | pathlib.Path):
            input_metadata (TensorMetadata): Mapping of input names to their data types and shapes.
            output_names (List[str]):
                    A list of output names of the model. This information is used by the
                    Comparator to determine which outputs to compare.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="tensorflow-trt-runner")
        self._model = model
        self.model = None

        self.input_metadata = TensorMetadata()
        for name, spec in input_metadata.items():
            self.input_metadata.add(name, spec.dtype, spec.shape)
        self.output_names = output_names

    def activate_impl(self):
        if isinstance(self._model, (str, Path)):
            self.model = tf.keras.models.load_model(str(self._model))
        else:
            self.model = self._model

    def deactivate_impl(self):
        self.model = None

    def get_input_metadata_impl(self):
        return self.input_metadata

    def infer_impl(self, feed_dict):

        start = time.time()
        if isinstance(self.model, tf.keras.Model):
            if isinstance(self.model._saved_model_inputs_spec, Mapping):
                outputs = self.model.predict(dict(zip(self.model._saved_model_inputs_spec.keys(), feed_dict.values())))
            else:
                outputs = self.model.predict(list(feed_dict.values()))

            validate_sample_output(outputs, Framework.TF2)
            if self.output_names is None:
                if isinstance(outputs, Mapping):
                    self.output_names = outputs.keys()
                else:
                    self.output_names = [f"output__{i}" for i in range(len(outputs))]

            if isinstance(outputs, numpy.ndarray):
                outputs = (outputs,)
            if isinstance(outputs, Mapping):
                outputs = outputs.values()
        else:
            sample = tuple(feed_dict.values())
            infer = self.model.signatures["serving_default"]
            inputs = dict(zip(self.model.signatures["serving_default"]._arg_keywords, sample))
            outputs = [output.numpy() for output in infer(**inputs).values()]
        end = time.time()

        out_dict = OrderedDict()
        for name, output in zip(self.output_names, outputs):
            out_dict[name] = output
        self.inference_time = end - start
        return out_dict
