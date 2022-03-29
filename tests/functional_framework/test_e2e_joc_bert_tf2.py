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

import argparse
import tempfile
from pathlib import Path

import input_pipeline  # pytype: disable=import-error
import squad_lib  # pytype: disable=import-error
import tensorflow as tf  # pytype: disable=import-error
import tokenization  # pytype: disable=import-error
from official.nlp import bert_modeling as modeling  # pytype: disable=import-error
from official.nlp import bert_models  # pytype: disable=import-error
from official.utils.misc import distribution_utils  # pytype: disable=import-error

import model_navigator as nav

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training, use_horovod):
    """Gets a closure to create a dataset.."""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="The BERT model config")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--vocab_file", type=str, default=None, required=True, help="Vocabulary mapping/file BERT was pretrainined on"
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Checkpoint path.")
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:

        navigator_workdir = Path(args.workdir)

        bert_config = modeling.BertConfig.from_json_file(args.config_file)
        tokenizer = tokenization.FullTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)

        eval_examples = squad_lib.read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative
        )

        tf_record_fn = Path(tmp_dir) / "eval.tf_record"
        eval_writer = squad_lib.FeatureWriter(filename=tf_record_fn.as_posix(), is_training=False)
        eval_features = []

        def _append_feature(feature, is_padding):
            if not is_padding:
                eval_features.append(feature)
            eval_writer.process_feature(feature)

        kwargs = {
            "examples": eval_examples,
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "doc_stride": args.doc_stride,
            "max_query_length": args.max_query_length,
            "is_training": False,
            "output_fn": _append_feature,
            "batch_size": args.predict_batch_size,
        }

        dataset_size = squad_lib.convert_examples_to_features(**kwargs)
        eval_writer.close()

        predict_dataset_fn = get_dataset_fn(
            tf_record_fn.as_posix(), args.max_seq_length, args.predict_batch_size, is_training=False, use_horovod=False
        )

        dataloader = []
        for sample, _ in iter(predict_dataset_fn()):
            dataloader.append({k: v for k, v in sample.items() if k != "unique_ids"})
            if len(dataloader) > 10:
                break

        with distribution_utils.get_strategy_scope(None):
            squad_model, _ = bert_models.squad_model(bert_config, args.max_seq_length, float_type=tf.float32)

        if args.checkpoint_path:
            checkpoint = tf.train.Checkpoint(model=squad_model)
            checkpoint.restore(args.checkpoint_path).expect_partial()

        squad_model(dataloader[0])

        pkg_desc = nav.tensorflow.export(
            model=squad_model,
            model_name="TF2-BERT",
            workdir=navigator_workdir,
            dataloader=dataloader,
            sample_count=10,
            target_formats=(nav.Format.TENSORRT,),
            target_precisions=(nav.TensorRTPrecision.FP32,),
            opset=13,
        )
        expected_formats = ("tf-trt-fp32", "onnx")
        for format, runtimes_status in pkg_desc.get_formats_status().items():
            for runtime, status in runtimes_status.items():
                assert (status == nav.Status.OK) == (
                    format in expected_formats
                ), f"{format} {runtime} status is {status}, but expected formats are {expected_formats}."

    nav.LOGGER.info("All models passed.")
