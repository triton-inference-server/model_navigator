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
# pytype: disable=import-error
import tempfile
from pathlib import Path

import model_navigator as nav
from model_navigator.converter.config import TensorRTPrecision

# pytype: enable=import-error


def check_model_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not Path(model_dir / "config.yaml").is_file():
        return False
    if not Path(model_dir / "model.savedmodel").exists():
        return False
    return True


def test_tf2_export_hf_distilbert():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_name = "distilbert-base-uncased"

        workdir = Path(tmp_dir) / "navigator_workdir"
        package_dir = workdir / f"{model_name}.nav"
        status_file = package_dir / "status.yaml"
        model_input_dir = package_dir / "model_input"
        model_output_dir = package_dir / "model_output"
        navigator_log_file = package_dir / "navigator.log"

        # pytype: disable=not-callable # TODO why is not-calleble being raised by pytype?
        nav.contrib.huggingface.tensorflow.export(
            model_name=model_name, override_workdir=True, workdir=workdir, target_precisions=(TensorRTPrecision.FP32,)
        )
        # pytype: enable=not-callable

        assert status_file.is_file()
        assert model_input_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_input_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert model_output_dir.is_dir()
        assert all(
            [path.suffix == ".npz" for samples_dir in model_output_dir.iterdir() for path in samples_dir.iterdir()]
        )
        assert navigator_log_file.is_file()

        # Passed formats
        assert check_model_dir(model_dir=package_dir / "tf-savedmodel")
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp32")

        # Failed formats
        assert check_model_dir(model_dir=package_dir / "tf-trt-fp16") is False
