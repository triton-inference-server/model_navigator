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
.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
PIP_INSTALL := pip install --extra-index-url https://pypi.ngc.nvidia.com

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-docs: ## remove test and coverage artifacts
	rm -rf docs/api
	$(MAKE) -C docs clean

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .pytype/

lint: ## check style with flake8 and pytype
	tox -e pytype,flake8

test: ## run tests quickly with the default Python
	pytest --ignore-glob='tests/test_*_pyt.py' --ignore-glob='tests/test_*_tf1.py' --ignore-glob='tests/test_*_tf2.py' --ignore-glob='tests/functional'

test-all: ## run tests on every Python version with tox
	tox

test-fw: ## run tests on framework containers
	docker run --gpus 0 --rm -it -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:21.03-py3 bash -c "pip install tox && tox -e pyt; make clean"
	docker run --gpus 0 --rm -it -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:21.03-tf2-py3 bash -c "pip install tox && tox -e tf2; make clean"
	#docker run --gpus 0 --rm -it -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash -c "pip install tox && tox -e tf1 && make clean"

test-func-e2e:
	@for f in $(shell ls ./tests/functional/run_test_e2e*.sh); do PYTHONPATH=$$PWD $${f}; done

test-func-convert:
	@for f in $(shell ls ./tests/functional/run_test_convert*.sh); do PYTHONPATH=$$PWD $${f}; done

coverage: ## check code coverage quickly with the default Python
	coverage run --source model_navigator -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc -f -o docs/api model_navigator
	$(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	$(BROWSER) docs/_build/html/index.html
	watchmedo shell-command -p '*.rst;*.md;*.py' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	# twine upload dist/*
	echo Commented out to avoid unfinished project publishing

dist: clean ## builds source and wheel package
	python -m build --sdist --wheel -o dist
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	$(PIP_INSTALL) .

install-with-cloud-extras: clean
	$(PIP_INSTALL) --upgrade --upgrade-strategy only-if-needed .[cloud]

install-with-framework-extras: clean
ifeq ($(origin TENSORFLOW_VERSION), undefined)
	$(PIP_INSTALL) --upgrade --upgrade-strategy only-if-needed .[pyt]
else
	$(PIP_INSTALL) --upgrade --upgrade-strategy only-if-needed .[tf]
endif

install-dev: clean
	$(PIP_INSTALL) -e .
	$(PIP_INSTALL) -r dev_requirements.txt

docker: clean
	docker build --network host -t model-navigator:latest .
