# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for onnx utils.

Note:
     Those test do not execute the conversion.
     The tests are checking if correct paths are executed on input arguments.
"""
from model_navigator.frameworks.onnx.utils import get_onnx_io_names
from tests.utils import get_assets_path


def test_get_onnx_input_metadata_return_filled_metadata_when_successfully_read_from_file(mocker):
    assets_path = get_assets_path()
    model_path = assets_path / "models" / "identity.onnx"

    input_names, output_names = get_onnx_io_names(model_path)
    assert len(input_names) == 1
    assert len(output_names) == 1

    assert input_names[0] == "X"
    assert output_names[0] == "Y"
