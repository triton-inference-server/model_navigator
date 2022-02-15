<!--
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

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

- to run tests locally or in [docker container](docs/installation.md#using-docker-container) you need to install project inside your current venv with all required development packages.
  To do that you can use `make install-dev` command.

## Running functional tests
- tests defined in `tests/functional` might be run with bash scripts
  - `tests/functional/run_test_e2e*.sh` executes all steps of pipeline: source model preparation, random dataset generation, conversion, analysis, helm chart generation
  - `tests/functional/run_test_convert*.sh` executes conversion steps of pipeline: source model preparation and conversion
- to run all e2e functional tests run `make test-func-e2e`
- to run all functional tests of conversion steps run `make test-func-convert`
- to run specific model use for example:
  ```
  PYTHONPATH=$PWD ./tests/functional/run_test_e2e_pytorch_vision_models.sh \
      ./tests/functional/pytorch_vision_models/e2e_config_pytorch_vision_resnet50_trace.yaml
  ```
- tests leave artifacts inside `$PWD/workspace` directory which might be reviewed manually.

**Note** If you run the tests from inside of container run `make install` to handle paths in tests correctly.

## Running unit tests
- tests defined inside `tests/unit/test_*.py` are run with pytest
  - `tests/unit/test_*_pyt.py` are run inside PyTorch container
  - `tests/unit/test_*_tf1.py` are run inside Tensorflow1 container
  - `tests/unit/test_*_tf2.py` are run inside Tensorflow2 container
  - rest is run inside `python:<version>` container
- to run tests locally inside your current venv run `make test`
- to run linters run `make lint`
- to run framework tests run `make test-fw`
- `make clean` cleans all tests related artifacts/caches etc
  - to clean directory from artifacts created during `make test-fw` use `sudo make clean` command
