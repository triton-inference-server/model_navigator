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
import ctypes
import uuid
from unittest.mock import patch

import pytest

from model_navigator.exceptions import ModelNavigatorException
from model_navigator.utils import devices


class MockCuda:
    err_str = ctypes.c_char_p(b"mock error")

    def __init__(self, count):
        self.devices = [uuid.UUID(int=i + 10) for i in range(count)]
        self.device_names = [f"device_{i}".encode() for i in range(count)]

    def cuInit(self, _):
        return 0

    def cuDeviceGetName(self, buf, size, i):
        try:
            i = i.value
        except AttributeError:
            pass
        if not 0 <= i < len(self.devices):
            return 1  # random error
        name = self.device_names[i]
        to_copy = min(size, len(name))
        buf[0:to_copy] = name[0:to_copy]
        return 0

    def cuDeviceGetCount(self, count):
        count._obj.value = len(self.devices)
        return 0

    def cuDeviceGetUuid(self, buf, i):
        try:
            i = i.value
        except AttributeError:
            pass
        if not 0 <= i < len(self.devices):
            return 1  # random error
        buf[:] = self.devices[i].bytes
        return 0

    def cuGetErrorString(self, err, bufp):
        bufp._obj.value = self.err_str.value
        return 0


def test_no_gpus():
    for v in [[], None]:
        gpus = devices.get_gpus(v)
        assert len(gpus) == 0


def test_all_1():
    for v in [["all"]]:
        with patch("model_navigator.utils.devices.cuda", new=MockCuda(1)) as mock_cuda:
            gpus = devices.get_gpus(v)
            assert len(gpus) == 1
            assert gpus[0] == f"GPU-{str(mock_cuda.devices[0])}"


def test_uuid_1():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(1)) as mock_cuda:
        gpus = devices.get_gpus([f"GPU-{str(mock_cuda.devices[0])}"])
        assert len(gpus) == 1
        assert gpus[0] == f"GPU-{str(mock_cuda.devices[0])}"


def test_some_8():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(8)) as mock_cuda:
        for gs in [[0, 1, 2, 3], [0, 2, 4, 6], [7, 5, 3, 1], [3, 5, 7, 1], list(range(8)), list(reversed(range(8)))]:
            gpus = devices.get_gpus([str(x) for x in gs])
            assert len(gpus) == len(gs)
            for i, g in enumerate(gs):
                assert gpus[i] == f"GPU-{str(mock_cuda.devices[g])}"


def test_some_uuid_8():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(8)) as mock_cuda:
        for gs in [[0, 1, 2, 3], [0, 2, 4, 6], [7, 5, 3, 1], [3, 5, 7, 1], list(range(8)), list(reversed(range(8)))]:
            gpus = devices.get_gpus([f"GPU-{mock_cuda.devices[g]}" for g in gs])
            assert len(gpus) == len(gs)
            for i, g in enumerate(gs):
                assert gpus[i] == f"GPU-{mock_cuda.devices[g]}"


def test_all_8():
    for v in [["all"]]:
        with patch("model_navigator.utils.devices.cuda", new=MockCuda(8)) as mock_cuda:
            gpus = devices.get_gpus(v)
            assert len(gpus) == 8
            for i in range(8):
                assert gpus[i] == f"GPU-{str(mock_cuda.devices[i])}"


def test_oob_8():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(8)):
        with pytest.raises(ModelNavigatorException):
            devices.get_gpus(["9"])


def test_invalid_uuid_8():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(8)):
        with pytest.raises(ModelNavigatorException):
            devices.get_gpus([f"GPU-{str(uuid.uuid4())}"])


def test_twice():
    with patch("model_navigator.utils.devices.cuda", new=MockCuda(1)) as mock_cuda:
        gpus = devices.get_gpus(gpus=[0])
        assert len(gpus) == 1
        assert gpus[0] == f"GPU-{str(mock_cuda.devices[0])}"

        gpus = devices.get_gpus(gpus=[0])
        assert len(gpus) == 1
        assert gpus[0] == f"GPU-{str(mock_cuda.devices[0])}"
