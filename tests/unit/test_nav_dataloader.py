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
import pathlib
import tempfile

import numpy as np
import pytest

from model_navigator.converter.dataloader import NavPackageDataloader
from model_navigator.utils.nav_package import NavPackageDirectory


@pytest.fixture
def nav_package():
    with tempfile.TemporaryDirectory(suffix=".nav") as package:
        path = pathlib.Path(package)
        inp_path = path / "model_input" / "test"
        inp_path.mkdir(parents=True)
        for i in range(4):
            for j in range(70):
                with open(inp_path / f"{100*i + j}.npz", "wb") as f:
                    np.savez(
                        f, input__0=np.arange(i * 4, dtype=np.int32), input__1=np.random.random((2 * i + 1, 2 * i + 5))
                    )
        yield NavPackageDirectory(path)


def test_batch_dim(nav_package):
    for bs in (1, 3, 50, 64):
        dataloader = NavPackageDataloader(nav_package, "test", max_batch_size=bs)
        sizes = set()
        for _i, sample in enumerate(dataloader):
            assert "input__0" in sample
            assert sample["input__0"].dtype == np.int32
            assert "input__1" in sample
            assert sample["input__1"].dtype == np.float
            for v in sample.values():
                assert v.shape[0] <= bs
                sizes.add(v.shape[0])
        assert bs in sizes
        assert 1 in sizes
        assert len(sizes) in (1, 2)
        assert _i > 0

        assert all(dataloader.max_shapes[n][0] == bs for n in dataloader.max_shapes)
        assert all(dataloader.min_shapes[n][0] == 1 for n in dataloader.min_shapes)
        assert dataloader.min_shapes.keys() == dataloader.max_shapes.keys()
        for n in dataloader.max_shapes:
            for a, b in zip(dataloader.max_shapes[n], dataloader.min_shapes[n]):
                assert a >= b
