# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

# pytype: disable=import-error
import tensorflow as tf

from model_navigator.tensor import NPTensorUtils, TensorUtils

# pytype: enable=import-error


def test_tf2_eq():
    with tf.device("CPU:0"):
        a = tf.constant([1, 1, 1], dtype=tf.float32)
        utils = TensorUtils.for_data(a)
        assert utils.eq(a, a)

        b = tf.Variable(a)
        assert utils.eq(a, b)

    with tf.device("GPU:0"):
        # not equal due to different devices
        c = tf.Variable(a)
        assert not utils.eq(a, c)

    # simple change
    d = tf.constant([0, 1, 1], dtype=tf.float32)
    assert not utils.eq(a, d)

    # different dtypes should return False
    e = tf.constant([1, 1, 1], dtype=tf.float64)
    assert not utils.eq(a, e)

    # different shapes also should return False
    f = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    assert not utils.eq(a, f)


def test_tf2_to_numpy():
    with tf.device("GPU:0"):
        a = tf.ones(4, dtype=tf.float32)
        utils = TensorUtils.for_data(a)
        b = utils.to_numpy(a)
        assert NPTensorUtils.eq(b, np.ones(4, dtype=np.float32))

        a = tf.Variable([np.nan, 1, 1, 1], dtype=tf.float32)
        c = utils.to_numpy(a)
        c_expected = np.array([np.nan, 1, 1, 1], dtype=np.float32)
        assert NPTensorUtils.eq(c, c_expected)
