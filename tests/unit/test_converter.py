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
import tempfile
from pathlib import Path

from model_navigator.converter import ConversionConfig, Converter
from model_navigator.converter.dataloader import Dataloader
from model_navigator.model import Format, ModelConfig
from model_navigator.results import State
from model_navigator.utils import Workspace


class MockDataloader(Dataloader):
    def __iter__(self):
        pass

    @property
    def max_shapes(self):
        return None

    @property
    def min_shapes(self):
        return None

    @property
    def opt_shapes(self):
        return None

    @property
    def dtypes(self):
        return None


def test_converter_return_src_model_if_it_matches_conversion_set():
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "navigator_workspace"
        workspace = Workspace(workspace_dir)
        converter = Converter(workspace=workspace, verbose=True)

        src_model = ModelConfig(model_name="MyModel", model_path=Path("tests/files/models/identity.savedmodel"))
        conversion_config = ConversionConfig(target_format=Format.TF_SAVEDMODEL)
        mock_dataloader = MockDataloader()
        conversion_results = converter.convert(
            src_model=src_model, conversion_config=conversion_config, dataloader=mock_dataloader
        )
        conversion_results = list(conversion_results)
        assert len(conversion_results) == 1
        result = conversion_results[0]
        assert result.status.state == State.SUCCEEDED, result.status.message
        assert result.output_model.path == src_model.model_path

        src_model = ModelConfig(model_name="MyModel", model_path=Path("tests/files/models/identity.onnx"))
        conversion_config = ConversionConfig(target_format=Format.ONNX)
        conversion_results = converter.convert(
            src_model=src_model, conversion_config=conversion_config, dataloader=mock_dataloader
        )
        conversion_results = list(conversion_results)
        assert len(conversion_results) == 1
        result = conversion_results[0]
        assert result.status.state == State.SUCCEEDED, result.status.message
        assert result.output_model.path == src_model.model_path

        src_model = ModelConfig(model_name="MyModel", model_path=Path("tests/files/models/identity.traced.pt"))
        conversion_config = ConversionConfig(target_format=Format.TORCHSCRIPT)
        conversion_results = converter.convert(
            src_model=src_model, conversion_config=conversion_config, dataloader=mock_dataloader
        )
        conversion_results = list(conversion_results)
        assert len(conversion_results) == 1
        result = conversion_results[0]
        assert result.status.state == State.SUCCEEDED, result.status.message
        assert result.output_model.path == src_model.model_path
