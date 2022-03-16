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
import pytest

from model_navigator.exceptions import ModelNavigatorInvalidPackageException
from model_navigator.utils.cli import select_input


# select_input tests
def test_select_one():
    inp = {"format": "torchscript", "path": "/torchscript", "status": "OK", "torch_jit": "script"}
    inp_onnx = {"format": "onnx", "path": "/x.onnx", "status": "OK"}
    assert select_input([inp]) == inp
    assert select_input([inp_onnx]) == inp_onnx


def test_select_prefer_ts_over_onnx():
    inp_ts = {"format": "torchscript", "path": "/x.ts", "status": "OK", "torch_jit": "script"}
    inp_onnx = {"format": "onnx", "path": "/x.onnx", "status": "OK"}
    assert select_input([inp_ts, inp_onnx]) == inp_ts
    assert select_input([inp_onnx, inp_ts]) == inp_ts


def test_select_prefer_ts_script_over_trace():
    inp_ts = {"format": "torchscript", "path": "/x.ts", "status": "OK", "torch_jit": "script"}
    inp_trace = {"format": "torchscript", "path": "/x.ts", "status": "OK", "torch_jit": "trace"}
    inp_onnx = {"format": "onnx", "path": "/x.onnx", "status": "OK"}
    assert select_input([inp_ts, inp_trace, inp_onnx]) == inp_ts
    assert select_input([inp_trace, inp_onnx, inp_ts]) == inp_ts
    assert select_input([inp_onnx, inp_ts, inp_trace]) == inp_ts


def test_select_prefer_savedmodel_over_onnx():
    inp_sm = {"format": "savedmodel", "path": "/x.savedmodel", "status": "OK"}
    inp_onnx = {"format": "onnx", "path": "/x.onnx", "status": "OK"}
    assert select_input([inp_sm, inp_onnx]) == inp_sm
    assert select_input([inp_onnx, inp_sm]) == inp_sm


def test_select_only_ok():
    inp_sm = {"format": "savedmodel", "status": {"Default": "FAIL"}}
    inp_onnx = {"format": "onnx", "path": "/x.onnx", "status": "OK"}
    assert select_input([inp_sm, inp_onnx]) == inp_onnx
    assert select_input([inp_onnx, inp_sm]) == inp_onnx


def test_empty():
    with pytest.raises(ModelNavigatorInvalidPackageException):
        select_input([])

    inp_sm = {"format": "savedmodel", "status": {"Default": "FAIL"}}
    inp_onnx = {"format": "onnx", "status": {"Default": "FAIL"}}
    with pytest.raises(ModelNavigatorInvalidPackageException):
        select_input([inp_onnx, inp_sm])

    with pytest.raises(ModelNavigatorInvalidPackageException):
        select_input([inp_sm, inp_onnx])
