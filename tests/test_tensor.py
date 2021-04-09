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
import tempfile
from pathlib import Path

import numpy as np
import pytest

from model_navigator.tensor import IOSpec, TensorSpec, TensorUtils


def test_numpy_eq():
    a = np.ones((8, 64), dtype=np.float32)
    utils = TensorUtils.for_data(a)
    assert utils.eq(a, a)

    # simple change
    b = a.copy()
    b[0, 0] = 0
    assert not utils.eq(a, b)

    # should also handle non-numeric values
    c = a.copy()
    c[0, 0] = np.nan
    assert not utils.eq(a, c)

    # different dtypes should return False
    d = a.copy()
    d = d.astype(np.float64)
    assert not utils.eq(a, d)

    # different shapes also should return False
    e = np.ones((9, 64))
    assert not utils.eq(a, e)


def test_numpy_to_numpy():
    a = np.ones((8, 64), dtype=np.float32)
    utils = TensorUtils.for_data(a)
    b = utils.to_numpy(a)
    assert utils.eq(a, b)
    assert a is b

    a[0, 0] = np.nan
    c = utils.to_numpy(a)
    assert utils.eq(a, c)
    assert a is c


def test_tensorspec():
    TensorSpec("name", shape=(-1, 224, 224, 3), dtype=np.dtype("float32"))
    with pytest.raises(TypeError):
        TensorSpec("name", shape=(-1, 224, 224, 3), dtype="float32")

    # pytype: disable=wrong-arg-types
    with pytest.raises(TypeError):
        TensorSpec("name", shape=[-1, 224, 224, 3], dtype=np.dtype("float32"))
    # pytype: enable=wrong-arg-types

    with pytest.raises(TypeError):
        TensorSpec("name", shape=(None, 224, 224, 3), dtype=np.dtype("float32"))


def test_tensorspec_without_dtype():
    spec = TensorSpec("name", shape=(-1, 224, 224, 3))
    assert spec.dtype is None


def test_iospec():
    IOSpec(
        {"tensor_in1": TensorSpec("tensor_in1", shape=(-1, 24, 24, 1), dtype=np.dtype("float32"))},
        {"tensor_out1": TensorSpec("tensor_out1", shape=(-1, 1000), dtype=np.dtype("float32"))},
    )


def test_write_read_iospec():
    io_spec = IOSpec(
        {"tensor_in1": TensorSpec("tensor_in1", shape=(-1, 24, 24, 1), dtype=np.dtype("float32"))},
        {"tensor_out1": TensorSpec("tensor_out1", shape=(-1, 1000), dtype=np.dtype("float32"))},
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "io_spec.yaml"
        io_spec.write(file_path)

        loaded_io_spec = IOSpec.from_file(file_path)
        assert io_spec == loaded_io_spec
