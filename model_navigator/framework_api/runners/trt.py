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

import numpy as np
from polygraphy.backend.trt import TrtRunner as _TrtRunner

from model_navigator.framework_api.logger import LOGGER


class TrtRunner(_TrtRunner):
    """
    Runs inference using PyTorch.
    """

    trt_casts = {np.dtype(np.int64): np.int32}

    def _cast_tensor(self, tensor):
        if tensor.dtype in self.trt_casts:
            LOGGER.debug(f"Casting f{tensor.dtype} tensor to f{self.trt_casts[tensor.dtype]}.")
            return tensor.astype(self.trt_casts[tensor.dtype])
        return tensor

    def infer(self, feed_dict, check_inputs=None, *args, **kwargs):
        feed_dict = {name: self._cast_tensor(tensor) for name, tensor in feed_dict.items()}
        return super().infer(feed_dict, check_inputs, *args, **kwargs)
