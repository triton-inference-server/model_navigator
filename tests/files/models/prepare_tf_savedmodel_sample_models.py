#!/usr/bin/env python3
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
import numpy as np
import tensorflow as tf  # pytype: disable=import-error


def prepare_identity(output_path: str):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x))
    model.predict(x=np.zeros((128, 3, 240, 240), dtype=np.float32))
    model.save(output_path)


def main():
    prepare_identity(output_path="tests/files/models/identity.savedmodel")


if __name__ == "__main__":
    main()
