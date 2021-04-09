<!--
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Testing

## Running tests
- tests defined inside `tests/test_*.py` are run with pytest
  - `tests/test_*_pyt.py` are run inside PyTorch container
  - `tests/test_*_tf1.py` are run inside Tensorflow1 container
  - `tests/test_*_tf2.py` are run inside Tensorflow2 container
  - rest is run inside `python:<version>` container
- to run tests locally you need to install project inside your current venv with all required development packages.
  To do that you can use `make install-dev` command
- to run tests locally inside your current venv run `make test`
- to run linters run `make lint`
- to run framework tests run `make test-fw`
- `make clean` cleans all tests related artifacts/caches etc
  - to clean directory from artifacts created during `make test-fw` use `sudo make clean` command
