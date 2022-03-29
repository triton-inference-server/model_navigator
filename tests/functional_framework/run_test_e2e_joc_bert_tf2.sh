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
BERT_PATH="${TEMPDIR}/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT/"

export BERT_PREP_WORKING_DIR="${BERT_PATH}/bert_prep"
mkdir -p ${BERT_PREP_WORKING_DIR}
export PYTHONPATH="${PYTHONPATH}:${BERT_PATH}"

git clone https://github.com/titipata/pubmed_parser ${TEMPDIR}/pubmed_parser

pip install ${TEMPDIR}/pubmed_parser
pip install \
  requests \
  tqdm \
  horovod \
  sentencepiece \
  tensorflow_hub \
  pynvml \
  wget \
  progressbar \
  git+https://github.com/NVIDIA/dllogger

python3 ${BERT_PATH}/data/bertPrep.py --action download --dataset squad
python3 ${BERT_PATH}/data/bertPrep.py --action download --dataset google_pretrained_weights

if [ -z "$1" ]
then
    WORKDIR=${TEMPDIR}
else
    WORKDIR=${1}
fi

./tests/functional_framework/test_e2e_joc_bert_tf2.py \
  --config_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json \
  --predict_file ${BERT_PREP_WORKING_DIR}/download/squad/v1.1/dev-v1.1.json \
  --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt \
  --checkpoint_path ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt.index \
  --workdir ${WORKDIR}

