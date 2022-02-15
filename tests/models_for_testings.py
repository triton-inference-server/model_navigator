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
# https://github.com/winnerineast/models-onnx/ - links to https servers
# https://github.com/onnx/models - uses Git LFS so harder to obtain during tests
# NVIDIA NGC models catalogue

ONNX_MODELS_URLS = {
    # "ResNet50": "https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz",
    "MobileNetV2": "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
}
