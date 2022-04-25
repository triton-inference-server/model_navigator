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

from polygraphy.backend.onnxrt import OnnxrtRunner as _OnnxrtRunner

from model_navigator.framework_api.exceptions import UserErrorContext
from model_navigator.framework_api.utils import Framework, validate_sample_output


class OnnxrtRunner(_OnnxrtRunner):
    def activate(self):
        with UserErrorContext():
            return super().activate()

    def infer(self, feed_dict, check_inputs=None, *args, **kwargs):
        feed_dict = {name: tensor for name, tensor in feed_dict.items() if name in self.get_input_metadata()}
        return super().infer(feed_dict, check_inputs, *args, **kwargs)

    def infer_impl(self, feed_dict):
        start = time.time()
        inference_outputs = self.sess.run(None, feed_dict)
        end = time.time()

        validate_sample_output(inference_outputs, Framework.ONNX)

        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), inference_outputs):
            out_dict[node.name] = out
        self.inference_time = end - start
        return out_dict
