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
from model_navigator.exceptions import ModelNavigatorInvalidPackageException


def select_input_format(models):
    """Automatically select which input model from the .nav package to use as input"""
    # sorted from most to least preferred
    PREFERRENCE_ORDER = [
        {"format": "torchscript", "torch_jit": "script"},
        {"format": "torchscript", "torch_jit": "trace"},
        {"format": "tf-savedmodel"},
        {"format": "onnx"},
    ]
    for fmt in PREFERRENCE_ORDER:
        for mod in models:
            if fmt.items() <= mod.items() and mod.get("path") is not None:
                return mod

    mod_list = "\n".join(f"{mod['format']}, path: {mod['path']}" for mod in models if "path" in mod)
    msg = "No valid models found in package."
    if mod_list:
        msg += f" Models found: {mod_list}"
    raise ModelNavigatorInvalidPackageException(msg)


class NavPackage:
    def __init__(self, path):
        self.path = path

    @property
    def datasets(self):
        return {path.name: list(path.glob("*.npz")) for path in self.path.glob("model_input/*")}

    def open(self, fpath):
        return open(self.path / fpath, "rb")

    @property
    def all_files(self):
        return (p.relative_to(self.path) for p in self.path.glob("**/*"))
