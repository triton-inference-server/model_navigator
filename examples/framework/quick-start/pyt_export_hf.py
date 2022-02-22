import argparse
import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import model_navigator.framework_api as nav

LOGGER = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser(description="Export huggingface models.")
    parser.add_argument("--workspace", type=str, default="./", help="Workspace path")
    parser.add_argument("--report-file", default="report.csv", type=str, help="Report file name (csv)")
    return parser.parse_args()


def get_model_names():
    return [
        "gpt2",
        # "cardiffnlp/twitter-roberta-base-sentiment",
        "bert-base-uncased",
        "distilgpt2",
        "distilbert-base-uncased",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        # "deepset/roberta-base-squad2",
        # "roberta-base",
        "bert-base-chinese",
    ]


if __name__ == "__main__":
    args = get_args()
    model_names = get_model_names()

    results = []
    for model_name in model_names:
        LOGGER.info(f"Exporting {model_name}...")
        try:

            nav_workdir = Path(args.workspace) / "navigator_workdir"
            artifacts_library = nav.huggingface.torch.export(
                model_name=model_name,
                dataset_name="imdb",
                override_workdir=True,
                keep_workdir=True,
                workdir=nav_workdir,
            )

            data = artifacts_library.get_formats_status()
            latency_status = artifacts_library.get_formats_latency()

            for format, latency in latency_status.items():
                if latency is not None:
                    data[format] = latency
            results.append({"model_name": model_name, **data, "comment": ""})

        except Exception as e:

            LOGGER.error(str(e))
            results.append({"model_name": model_name, "comment": str(e)})

    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="psql"))
    df.to_csv(args.report_file, index=False)
