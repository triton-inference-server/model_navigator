# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import torch  # pytype: disable=import-error
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata


class PytRunner(BaseRunner):
    """
    Runs inference using PyTorch.
    """

    def __init__(self, model, input_metadata, output_names, name=None, forward_kw_names=None):
        """
        Args:
            model (Union[torch.nn.Module, Callable() -> torch.nn.Module]):
                    A torch.nn.Module or subclass or a callable that returns one.
            input_metadata (Dict[str, TensorSpec]): Mapping of input names to their specs (data types and shapes).
            output_names (List[str]):
                    A list of output names of the model. This information is used by the
                    Comparator to determine which outputs to compare.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="pytorch-runner")
        self.model = model

        self.input_metadata = TensorMetadata()
        for name, spec in input_metadata.items():
            self.input_metadata.add(name, spec.dtype, spec.shape)
        self.output_names = output_names
        self._forward_kw_names = forward_kw_names

    def activate_impl(self):
        self.model.eval()

    def get_input_metadata_impl(self):
        return self.input_metadata

    def infer_impl(self, feed_dict):
        with torch.no_grad():
            inputs = [
                torch.from_numpy(val.astype(dtype)).cuda()
                for (val, (dtype, _)) in zip(feed_dict.values(), self.input_metadata.values())
            ]
            if self._forward_kw_names is None:
                start = time.time()
                outputs = self.model(*inputs)
                end = time.time()
            else:
                inputs_dict = dict(zip(self._forward_kw_names, inputs))
                start = time.time()
                outputs = self.model(**inputs_dict)
                end = time.time()

        out_dict = OrderedDict()
        for name, output in zip(self.output_names, outputs):
            out_dict[name] = output.cpu().numpy()
        self.inference_time = end - start
        return out_dict

    def deactivate_impl(self):
        del self.model
