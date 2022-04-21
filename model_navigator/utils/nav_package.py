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
import abc
import logging
import pathlib
import zipfile
from collections import defaultdict
from typing import IO, Dict, Iterable, Union

from model_navigator.exceptions import ModelNavigatorInvalidPackageException

LOGGER = logging.getLogger(__name__)


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


class NavPackage(abc.ABC):
    def __init__(self, path):
        self._path = pathlib.Path(path)

    @abc.abstractproperty
    def datasets(self) -> Dict[str, Iterable[str]]:
        ...

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @abc.abstractmethod
    def open(self, fpath: Union[str, pathlib.Path]) -> IO[bytes]:
        ...

    @abc.abstractproperty
    def all_files(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def vfs_path_to_member(self, member_path: Union[str, pathlib.Path]) -> pathlib.Path:
        ...


class NavPackageDirectory(NavPackage):
    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)

    @property
    def datasets(self):
        return {path.name: [p.as_posix() for p in path.glob("*.npz")] for path in self.path.glob("model_input/*")}

    def open(self, fpath: Union[str, pathlib.Path]) -> IO[bytes]:
        return open(self.path / fpath, "rb")

    @property
    def all_files(self):
        return (p.relative_to(self.path).as_posix() for p in self.path.glob("**/*"))

    def vfs_path_to_member(self, member_path: Union[str, pathlib.Path]):
        return self.path / member_path


class ZippedNavPackage(NavPackage):
    def __init__(self, path: Union[str, pathlib.Path]):
        super().__init__(path)
        # note: the zipfile never gets closed. We assume only a few such
        # objects will be created
        self.arc = zipfile.ZipFile(self.path, "r")

    @property
    def datasets(self):
        all_files = self.arc.namelist()
        result = defaultdict(list)
        for f in all_files:
            p = pathlib.Path(f)
            try:
                if p.parts[0] != "model_input":
                    continue
                if len(p.parts) > 2:
                    result[p.parts[1]].append(p.as_posix())
            except IndexError:
                pass

        return result

    def open(self, fpath: Union[str, pathlib.Path]) -> IO[bytes]:
        return self.arc.open(str(fpath), "r")

    @property
    def all_files(self):
        return self.arc.namelist()

    def vfs_path_to_member(self, member_path: Union[str, pathlib.Path]):
        raise NotImplementedError()

    def __getstate__(self):
        dct = dict(self.__dict__)
        del dct["arc"]
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.arc = zipfile.ZipFile(self.path, "r")


def from_path(path: Union[str, pathlib.Path]):
    for cls in [ZippedNavPackage, NavPackageDirectory]:
        try:
            return cls(path)
        except Exception:
            pass
    else:
        raise ModelNavigatorInvalidPackageException("Unrecognized package format")
