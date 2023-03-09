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
"""Tests collecting input metadata from ONNX file."""
import json
import pathlib
import tempfile

import pytest
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

from model_navigator.commands.convert.onnx.collect_onnx_input_metadata import collect_onnx_input_metadata
from tests.utils import get_assets_path


def test_collect_onnx_input_metadata_raise_onnxruntime_exception_when_file_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        results_path = tmpdir / "results.json"

        assets_path = get_assets_path()
        model_path = assets_path / "models" / "notexists.onnx"

        with pytest.raises(NoSuchFile):
            collect_onnx_input_metadata(
                model_path=model_path.as_posix(),
                input_metadata=[{"name": "X", "shape": (-1, 3, 8, 8), "dtype": "float32"}],
                output_metadata=[{"name": "Y", "shape": (-1, 3, 8, 8), "dtype": "float32"}],
                results_path=results_path.as_posix(),
            )

        assert results_path.is_file() is False


def test_collect_onnx_input_metadata_save_inputs_to_json_file_when_metadata_obtained():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        results_path = tmpdir / "results.json"

        assets_path = get_assets_path()
        model_path = assets_path / "models" / "identity.onnx"

        collect_onnx_input_metadata(
            model_path=model_path.as_posix(),
            input_metadata=[{"name": "X", "shape": (-1, 3, 8, 8), "dtype": "float32"}],
            output_metadata=[{"name": "Y", "shape": (-1, 3, 8, 8), "dtype": "float32"}],
            results_path=results_path.as_posix(),
        )

        assert results_path.is_file() is True

        with results_path.open("r") as fp:
            input_metadata = json.load(fp)

        assert input_metadata == [{"name": "X", "shape": [-1, 3, -1, -1], "dtype": "float32"}]
