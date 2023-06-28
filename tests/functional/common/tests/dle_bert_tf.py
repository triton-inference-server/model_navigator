# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
"""e2e tests for exporting BERT TensorFlow model from Deep Learning Examples"""
import logging
import os
import pathlib
import sys
import tempfile
from typing import Optional, Tuple

import model_navigator as nav

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{version}-tf2-py3",
    "repository": "https://github.com/NVIDIA/DeepLearningExamples",
    "model_dir": "TensorFlow2/LanguageModeling/BERT/",
}


def _get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training, use_horovod):
    """Gets a closure to create a dataset.."""
    import input_pipeline  # pytype: disable=import-error

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed BERT pretraining."""
        batch_size = ctx.get_per_replica_batch_size(global_batch_size) if ctx else global_batch_size
        dataset = input_pipeline.create_squad_dataset(
            input_file_pattern,
            max_seq_length,
            batch_size,
            is_training=is_training,
            input_pipeline_context=ctx,
            use_horovod=use_horovod,
        )
        return dataset

    return _dataset_fn


def _dle_bert_tf_test(
    config_file: pathlib.Path,
    vocab_file: pathlib.Path,
    predict_file: pathlib.Path,
    checkpoint_file: pathlib.Path,
    batch_size: int = 8,
    max_seq_length: int = 128,
    doc_stride: int = 128,
    max_query_length: int = 64,
    version_2_with_negative: bool = False,
    do_lower_case: bool = False,
    max_batch_size: Optional[int] = None,
    input_names: Optional[Tuple] = None,
    trt_profile: Optional[nav.TensorRTProfile] = None,
):
    import squad_lib  # pytype: disable=import-error
    import tensorflow as tf  # pytype: disable=import-error
    import tokenization  # pytype: disable=import-error
    from official.nlp import bert_modeling as modeling  # pytype: disable=import-error
    from official.nlp import bert_models  # pytype: disable=import-error
    from official.utils.misc import distribution_utils  # pytype: disable=import-error

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    bert_config = modeling.BertConfig.from_json_file(config_file)
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)

    eval_examples = squad_lib.read_squad_examples(
        input_file=predict_file, is_training=False, version_2_with_negative=version_2_with_negative
    )

    tf_record_fn = pathlib.Path.cwd() / "eval.tf_record"
    eval_writer = squad_lib.FeatureWriter(filename=tf_record_fn.as_posix(), is_training=False)
    eval_features = []

    def _append_feature(feature, is_padding):
        if not is_padding:
            eval_features.append(feature)
        eval_writer.process_feature(feature)

    kwargs = {
        "examples": eval_examples,
        "tokenizer": tokenizer,
        "max_seq_length": max_seq_length,
        "doc_stride": doc_stride,
        "max_query_length": max_query_length,
        "is_training": False,
        "output_fn": _append_feature,
        "batch_size": batch_size,
    }

    squad_lib.convert_examples_to_features(**kwargs)
    eval_writer.close()

    predict_dataset_fn = _get_dataset_fn(
        tf_record_fn.as_posix(), max_seq_length, batch_size, is_training=False, use_horovod=False
    )

    dataloader = []
    for sample, _ in iter(predict_dataset_fn()):
        dataloader.append({k: v for k, v in sample.items() if k != "unique_ids"})
        if len(dataloader) > 10:
            break

    with distribution_utils.get_strategy_scope(None):
        squad_model, _ = bert_models.squad_model(bert_config, max_seq_length, float_type=tf.float32)

    if checkpoint_file.exists():
        checkpoint = tf.train.Checkpoint(model=squad_model)
        checkpoint.restore(checkpoint_file).expect_partial()

    squad_model(dataloader[0])

    package = nav.tensorflow.optimize(
        model=squad_model,
        dataloader=dataloader,
        verbose=True,
        optimization_profile=nav.OptimizationProfile(max_batch_size=max_batch_size),
        input_names=input_names,
        custom_configs=(
            nav.OnnxConfig(opset=13),
            nav.TensorRTConfig(trt_profile=trt_profile),
            nav.TensorFlowTensorRTConfig(trt_profile=trt_profile),
        ),
    )

    return package


def dle_bert_tf(
    model_dir: str,
    git_url: str,
    max_batch_size: Optional[int] = None,
    input_names: Optional[Tuple] = None,
    trt_profile: Optional[nav.TensorRTProfile] = None,
):
    from git import Repo

    from tests import utils

    with tempfile.TemporaryDirectory() as tmp:
        repo = pathlib.Path(tmp)
        Repo.clone_from(git_url, repo)

        pubmed_parser_dir = repo / "pubmed_parser"
        Repo.clone_from("https://github.com/titipata/pubmed_parser", pubmed_parser_dir)

        model_dir = repo / model_dir
        bert_prep_dir = model_dir / "bert_prep"
        os.environ["BERT_PREP_WORKING_DIR"] = bert_prep_dir.as_posix()

        commands = [
            f"pip install {pubmed_parser_dir}",
            "python data/bertPrep.py --action download --dataset squad",
            "python data/bertPrep.py --action download --dataset google_pretrained_weights",
        ]
        for command in commands:
            utils.exec_command(command, workspace=model_dir.as_posix(), shell=True)

        sys.path.append(model_dir.as_posix())
        os.chdir(model_dir.as_posix())

        package = _dle_bert_tf_test(
            config_file=bert_prep_dir / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json",
            predict_file=bert_prep_dir / "download/squad/v1.1/dev-v1.1.json",
            vocab_file=bert_prep_dir / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt",
            checkpoint_file=bert_prep_dir
            / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt.index",
            max_batch_size=max_batch_size,
            input_names=input_names,
            trt_profile=trt_profile,
        )

        return package
