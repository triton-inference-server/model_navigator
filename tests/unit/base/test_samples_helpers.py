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

import pathlib
import tempfile

import numpy
import pytest

from model_navigator.commands.data_dump.samples import _validate_tensor, samples_to_npz
from model_navigator.exceptions import ModelNavigatorUserInputError
from model_navigator.utils.dataloader import load_samples


def test_validate_tensor_raise_exception_when_values_are_nan():
    tensor = numpy.empty(shape=(3, 3))
    tensor[:] = numpy.NaN

    with pytest.raises(ModelNavigatorUserInputError):
        # model_navigator.exceptions.ModelNavigatorUserInputError:
        #   Tensor data contains `NaN` value. Please verify the dataloader and model.
        _validate_tensor(tensor=tensor, raise_on_error=True)

    _validate_tensor(tensor=tensor, raise_on_error=False)


def test_validate_tensor_raise_exception_when_values_are_inf():
    tensor = numpy.empty(shape=(3, 3))
    tensor[:] = numpy.inf

    with pytest.raises(ModelNavigatorUserInputError):
        # model_navigator.exceptions.ModelNavigatorUserInputError:
        #   Tensor data contains `inf` value. Please verify the dataloader and model.
        _validate_tensor(tensor=tensor, raise_on_error=True)

    _validate_tensor(tensor=tensor, raise_on_error=False)


def test_validate_tensor_not_raise_exception_when_correct_tensor_passed():
    tensor = numpy.full(shape=(3, 3), fill_value=3)

    _validate_tensor(tensor=tensor, raise_on_error=True)


def test_samples_to_npz_create_file_with_valid_samples_when_valid_tensors_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_filepath = pathlib.Path(tmpdir)
        input_name = "input_0"
        batch_dim = 0
        sample = {input_name: numpy.full(shape=(1, 3), fill_value=3)}
        samples = [sample for _ in range(0, 10)]
        samples_to_npz(samples=samples, path=sample_filepath, batch_dim=batch_dim)

        loaded_samples = []
        for npz in sample_filepath.iterdir():
            loaded_sample = {}
            with numpy.load(npz) as data:
                for k, v in data.items():
                    v = numpy.expand_dims(v, batch_dim)
                    loaded_sample[k] = v
            loaded_samples.append(loaded_sample)

        assert len(samples) == len(loaded_samples)
        for s, l_s in zip(samples, loaded_samples):
            assert len(s) == len(l_s)
            for (k1, v1), (k2, v2) in zip(s.items(), l_s.items()):
                assert k1 == k2
                assert (v1 == v2).all()


def test_samples_are_saved_and_loaded_in_the_same_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_filepath = pathlib.Path(tmpdir) / "model_input" / "correctness"

        input_name = "input_0"
        batch_dim = 0
        samples = []
        for fill_value in range(0, 10):
            sample = {input_name: numpy.full(shape=(1, 3), fill_value=fill_value)}
            samples.append(sample)
        samples_to_npz(samples=samples, path=sample_filepath, batch_dim=batch_dim)
        loaded_samples = load_samples(samples_name="correctness_samples", workspace=tmpdir, batch_dim=batch_dim)

        assert len(samples) == len(loaded_samples)
        for s, l_s in zip(samples, loaded_samples):
            assert len(s) == len(l_s)
            for (k1, v1), (k2, v2) in zip(s.items(), l_s.items()):
                assert k1 == k2
                assert (v1 == v2).all()
