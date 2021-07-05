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
import logging
import os
import pathlib
import shutil
import tarfile
import typing
import urllib.request
import zipfile
from tempfile import TemporaryDirectory
from typing import Any, Callable
from urllib.parse import urlparse

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class Prefixes:
    HTTP = ["http://", "https://"]
    GCS = ["gs://"]
    AWS = ["s3://"]
    AZURE = ["as://"]


def _download_progress(t: Any) -> Callable:
    last_downloaded = [0]

    def update(downloaded: int = 1, chunk_size: int = 1, total_size: int = None):
        if total_size not in (None, -1):
            t.total = total_size

        t.update((downloaded - last_downloaded[0]) * chunk_size)
        last_downloaded[0] = downloaded

    return update


def _http_downloader(file_uri: str, tmpdir) -> pathlib.Path:
    LOGGER.info(f"Downloading file from {file_uri}")

    parsed_url = urlparse(file_uri)
    filename = os.path.basename(parsed_url.path)

    with tqdm(unit="B") as t:
        urllib.request.urlretrieve(file_uri, filename=filename, reporthook=_download_progress(t))

    dst_path = pathlib.Path(tmpdir) / filename
    shutil.move(filename, dst_path.as_posix())
    LOGGER.info(f"File saved in {dst_path}")

    if not dst_path.is_file():
        raise RuntimeError(f"File {file_uri} was not downloaded correctly.")

    return dst_path


def _gcs_downloader(file_uri: str, tmpdir) -> pathlib.Path:
    LOGGER.info(f"Downloading file {file_uri} from Google Cloud Storage ")
    parts = file_uri[5:].split("/")
    bucket = parts[0]
    filename = parts[-1]
    resource = "/".join(parts[1:])

    LOGGER.info(f"Bucket: {bucket}")
    LOGGER.info(f"Filename: {filename}")

    from google.cloud import storage

    client = storage.Client()
    bucket = client.get_bucket(bucket)
    blob = storage.Blob(resource, bucket)
    tmp_file = pathlib.Path(tmpdir) / filename

    with open(tmp_file, "wb+") as f:
        blob.download_to_file(f)

    LOGGER.info(f"File saved in {tmp_file}")

    if not tmp_file.is_file():
        raise RuntimeError(f"File {file_uri} was not downloaded correctly.")

    return tmp_file


def _s3_downloader(file_uri: str, tmpdir) -> pathlib.Path:
    LOGGER.info(f"Downloading file {file_uri} from AWS S3")

    parts = file_uri[5:].split("/")
    bucket = parts[0]
    filename = parts[-1]
    resource = "/".join(parts[1:])

    import boto3

    s3 = boto3.client("s3")

    tmp_file = pathlib.Path(tmpdir) / filename
    s3.download_file(bucket, resource, tmp_file.as_posix())

    LOGGER.info(f"File saved in {tmp_file}")

    if not tmp_file.is_file():
        raise RuntimeError(f"File {file_uri} was not downloaded correctly.")

    return tmp_file


def _azure_downloader(file_uri: str, tmpdir) -> pathlib.Path:
    LOGGER.info(f"Downloading file {file_uri} from Azure Cloud Storage.")

    parts = file_uri[5:].split("/")
    container_name = parts[1]
    filename = parts[-1]
    resource = "/".join(parts[2:])

    from azure.storage.blob import BlobServiceClient

    blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])

    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        blob_client = container_client.get_blob_client(resource)

        tmp_file = pathlib.Path(tmpdir) / filename
        with open(tmp_file, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
    finally:
        container_client.delete_container()

    LOGGER.info(f"File saved in {tmp_file}")

    if not tmp_file.is_file():
        raise RuntimeError(f"File {file_uri} was not downloaded correctly.")

    return tmp_file


class Downloader:
    def __init__(self, file_uri: str, dst_path: typing.Union[str, pathlib.Path]):
        self._file_uri = file_uri
        self._dst_path = pathlib.Path(dst_path)
        self._downloader = self._get_downloader(file_uri=file_uri)

    def get_file(self) -> pathlib.Path:
        with TemporaryDirectory() as tmpdir:
            file_path = self._downloader(self._file_uri, tmpdir)
            if zipfile.is_zipfile(file_path) and file_path.suffix == ".zip":
                file_path = self._unzip(file_to_unpack=file_path)
            elif tarfile.is_tarfile(file_path) and ".tar" in file_path.suffix:
                file_path = self._untar(file_to_unpack=file_path)
            else:
                LOGGER.info(f"Moving file {file_path} to {self._dst_path.as_posix()}")
                shutil.move(file_path.as_posix(), self._dst_path.as_posix())

        return file_path

    def _get_downloader(self, file_uri: str):
        if self._file_uri_match_prefix(file_uri, prefixes=Prefixes.HTTP):
            return _http_downloader
        elif self._file_uri_match_prefix(file_uri, prefixes=Prefixes.GCS):
            return _gcs_downloader
        elif self._file_uri_match_prefix(file_uri, prefixes=Prefixes.AWS):
            return _s3_downloader
        elif self._file_uri_match_prefix(file_uri, prefixes=Prefixes.AZURE):
            return _azure_downloader

        raise ValueError(f"Unsupported resource {file_uri}.")

    def _file_uri_match_prefix(self, file_uri: str, *, prefixes: typing.List[str]):
        return any([file_uri.startswith(prefix) for prefix in prefixes])

    def _untar(self, file_to_unpack: pathlib.Path) -> pathlib.Path:
        with TemporaryDirectory() as tmpdir:
            LOGGER.info(f"Unpacking tar archive {file_to_unpack} to {tmpdir}"),
            tf = tarfile.TarFile.open(name=file_to_unpack, mode="r")
            tf.extractall(tmpdir)
            LOGGER.info("done")

            LOGGER.info(f"Removing tar file: {file_to_unpack}")
            os.remove(file_to_unpack.as_posix())
            LOGGER.info("done")

            items = [item for item in pathlib.Path(tmpdir).iterdir()]
            if len(items) > 1:
                raise ValueError("Too many files in archive. Expected single file/directory with model.")

            LOGGER.info(f"Moving file {items[0].as_posix()} to {self._dst_path.as_posix()}")
            shutil.move(items[0].as_posix(), self._dst_path.as_posix())

        return self._dst_path

    def _unzip(self, file_to_unpack: pathlib.Path) -> pathlib.Path:
        with TemporaryDirectory() as tmpdir:
            LOGGER.info(f"Unpacking zip archive {file_to_unpack} to {tmpdir}")
            with zipfile.ZipFile(file_to_unpack.as_posix(), "r") as zf:
                zf.extractall(tmpdir)
            LOGGER.info("done")

            LOGGER.info(f"Removing zip file: {file_to_unpack}")
            os.remove(file_to_unpack.as_posix())
            LOGGER.info("done")

            items = [item for item in pathlib.Path(tmpdir).iterdir()]
            if len(items) > 1:
                raise ValueError("Too many files in archive. Expected single file/directory with model.")

            LOGGER.info(f"Moving file {items[0].as_posix()} to {self._dst_path.as_posix()}")
            shutil.move(items[0].as_posix(), self._dst_path.as_posix())

        return self._dst_path
