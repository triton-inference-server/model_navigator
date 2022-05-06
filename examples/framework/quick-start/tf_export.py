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
# limitations under the License.2
import numpy
import tensorflow as tf

import model_navigator as nav

dataloader = [tf.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tf.dtypes.float32) for _ in range(10)]

inp = tf.keras.layers.Input((224, 224, 3))
layer_output = tf.keras.layers.Lambda(lambda x: x)(inp)
layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
layer_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
model_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
model = tf.keras.Model(inp, model_output)


pkg_desc = nav.tensorflow.export(
    model=model,
    dataloader=dataloader,
    override_workdir=True,
    target_formats=(nav.Format.ONNX,),
)

# Verify ONNX format against model in framework
sample_count = 100
valid_outputs = 0
for _ in range(sample_count):
    random_sample = tf.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)

    # Use source model to generate dummy ground truth
    gt = [model.predict(random_sample)]

    feed_dict = {"input__0": random_sample.numpy()}
    onnx_runner = pkg_desc.get_runner(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)
    with onnx_runner:
        output = onnx_runner.infer(feed_dict)

    # Compare output and gt
    for a, b in zip(gt, output.values()):
        if numpy.allclose(a, b, atol=0, rtol=0):
            valid_outputs += 1

accuracy = float(valid_outputs) / float(sample_count)
print(f"Accuracy: {accuracy}")

if accuracy > 0.8:
    pkg_desc.set_verified(format=nav.Format.ONNX, runtime=nav.RuntimeProvider.CUDA)

# Save nav package
pkg_desc.save("my_model.nav")
