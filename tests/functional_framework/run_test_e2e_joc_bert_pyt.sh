#!/usr/bin/env bash
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

set -ex

TEMPDIR=$(mktemp -d)

function cleanup {
    echo "Cleanup..."
    rm -r ${TEMPDIR}
}

trap cleanup EXIT

git clone https://github.com/NVIDIA/DeepLearningExamples ${TEMPDIR}/DeepLearningExamples
BERT_PATH="${TEMPDIR}/DeepLearningExamples/PyTorch/LanguageModeling/BERT/"

export BERT_PREP_WORKING_DIR="${BERT_PATH}/bert_prep"
mkdir -p ${BERT_PREP_WORKING_DIR}

export PYTHONPATH="${PYTHONPATH}:${BERT_PATH}"
pip install $(grep -v '^ *#\|^onnxruntime' ${BERT_PATH}/requirements.txt | grep .)

python3 ${BERT_PATH}/data/bertPrep.py --action download --dataset squad

./tests/functional_framework/test_e2e_joc_bert_pyt.py \
  --config_file ${BERT_PATH}/bert_configs/base.json \
  --predict_file ${BERT_PREP_WORKING_DIR}/download/squad/v1.1/dev-v1.1.json \
  --vocab_file ${BERT_PATH}/vocab/vocab
