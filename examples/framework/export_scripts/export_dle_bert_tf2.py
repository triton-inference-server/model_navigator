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
import os
import pathlib
import subprocess
import sys
import tempfile

import tensorflow as tf  # pytype: disable=import-error

import model_navigator as nav

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MODEL_NAMES = ["BERT"]

REQUIREMENTS = [
    "requests",
    "tqdm",
    "horovod",
    "sentencepiece",
    "tensorflow_hub",
    "pynvml",
    "wget",
    "progressbar",
    "git+https://github.com/NVIDIA/dllogger",
    "git+https://github.com/titipata/pubmed_parser",
]


def setup_env(workdir):
    cmd = [
        "git",
        "clone",
        "https://github.com/NVIDIA/DeepLearningExamples",
        (workdir / "DeepLearningExamples").as_posix(),
    ]
    subprocess.run(cmd, check=True)
    bert_path = workdir / "DeepLearningExamples/TensorFlow2/LanguageModeling/BERT/"
    sys.path.append(bert_path.as_posix())
    bert_prep_working_dir = bert_path / "bert_prep"
    bert_prep_working_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BERT_PREP_WORKING_DIR"] = bert_prep_working_dir.as_posix()

    cmd = [sys.executable, "-m", "pip", "install"] + REQUIREMENTS
    subprocess.run(cmd, check=True)

    subprocess.run(
        [sys.executable, (bert_path / "data/bertPrep.py").as_posix(), "--action", "download", "--dataset", "squad"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            (bert_path / "data/bertPrep.py").as_posix(),
            "--action",
            "download",
            "--dataset",
            "google_pretrained_weights",
        ],
        check=True,
    )

    return bert_path, bert_prep_working_dir


def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size, is_training, use_horovod):
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


def get_verification_status_dummy(runner):
    """Dummy verification function."""
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", type=str, choices=MODEL_NAMES)
    group.add_argument(
        "--list-models",
        action="store_true",
    )
    parser.add_argument(
        "--output-path",
        type=str,
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
    )
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
    return parser.parse_args()


def main():

    args = parse_args()
    if args.list_models:
        print(MODEL_NAMES)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        bert_path, bert_prep_working_dir = setup_env(pathlib.Path(tmpdir))

        import squad_lib  # pytype: disable=import-error
        import tokenization  # pytype: disable=import-error
        from official.nlp import bert_modeling as modeling  # pytype: disable=import-error
        from official.nlp import bert_models  # pytype: disable=import-error
        from official.utils.misc import distribution_utils  # pytype: disable=import-error

        config_file = (
            bert_prep_working_dir / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json"
        )
        predict_file = bert_prep_working_dir / "download/squad/v1.1/dev-v1.1.json"
        vocab_file = bert_prep_working_dir / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt"
        checkpoint_path = (
            bert_prep_working_dir / "download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_model.ckpt.index"
        )

        bert_config = modeling.BertConfig.from_json_file(config_file)
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=args.do_lower_case)

        eval_examples = squad_lib.read_squad_examples(
            input_file=predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative
        )

        tf_record_fn = pathlib.Path(tmpdir) / "eval.tf_record"
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

        squad_lib.convert_examples_to_features(**kwargs)
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

        checkpoint = tf.train.Checkpoint(model=squad_model)
        checkpoint.restore(checkpoint_path.as_posix()).expect_partial()

        squad_model(dataloader[0])

        pkg_desc = nav.tensorflow.export(
            model=squad_model,
            model_name=f"{args.model_name}_tf2",
            dataloader=dataloader,
            sample_count=10,
            opset=13,
            target_precisions="fp32",
        )

        for model_status in pkg_desc.navigator_status.model_status:
            for runtime_results in model_status.runtime_results:
                if runtime_results.status == nav.Status.OK:
                    runner = pkg_desc.get_runner(
                        format=model_status.format,
                        precision=model_status.precision,
                        runtime=runtime_results.runtime,
                    )
                    verified = get_verification_status_dummy(runner)
                    if verified:
                        pkg_desc.set_verified(
                            format=model_status.format,
                            precision=model_status.precision,
                            runtime=runtime_results.runtime,
                        )
                        nav.LOGGER.info(
                            f"{model_status.format=}, {model_status.precision=}, {runtime_results.runtime=} verified."
                        )
                    else:
                        nav.LOGGER.warning(
                            f"{model_status.format=}, {model_status.precision=}, {runtime_results.runtime=} not verified."
                        )

        output_path = args.output_path or f"{args.model_name}_tf2.nav"
        pkg_desc.save(output_path, keep_workdir=args.keep_workdir, override=True)


if __name__ == "__main__":
    main()
