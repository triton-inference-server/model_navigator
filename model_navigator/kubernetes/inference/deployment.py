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
import typing

from model_navigator.kubernetes.helm import Deployment as DeploymentConfig
from model_navigator.kubernetes.helm import Volume
from model_navigator.kubernetes.internals import Paths
from model_navigator.kubernetes.triton import TRITON_LOAD_MODE, TritonServer


def wrap(lst: typing.List, condition: str, required: typing.Optional[str] = None):
    lst.insert(
        0,
        {
            f"{{{{- if {condition} }}}}": None,
        },
    )
    if required:
        lst.append(
            {
                "{{ else }}": None,
                f'{{{{ required (printf "{required}") nil }}}}': None,
            }
        )

    lst.append(
        {
            "{{ end }}": None,
        }
    )


class Deployment(DeploymentConfig):
    @property
    def volumes(self):
        volumes = [
            Volume(
                name="model-repository",
                env="MODEL_REPOSITORY_PATH",
                path=Paths.MODEL_REPOSITORY_PATH,
                empty_dir={},
            ),
            Volume(name="shared-dir", env="SHARED_DIR", path=Paths.SHARED_DIR, empty_dir={}),
            Volume(name="shared-memory", env=None, path="/dev/shm", empty_dir={"medium": "Memory"}),
        ]

        return volumes

    def data(self):
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
        }

        metadata = {
            "name": self.name,
            "namespace": "{{ .Release.Namespace }}",
            "labels": {
                "app": '{{ template "selector.name" . }}',
                "chart": '{{ template "selector.chart" . }}',
                "release": "{{ .Release.Name }}",
                "heritage": "{{ .Release.Service }}",
            },
        }

        model_uri = [
            {
                "name": "MODEL_URI",
                "value": "{{ quote .Values.deployer.modelUri }}",
            }
        ]
        wrap(lst=model_uri, condition=".Values.deployer.modelUri", required="Model URI is required.")

        env = []
        env.extend(model_uri)

        volumeMounts = []
        volumeAttach = []

        self._prepare_volumes(env, volumeAttach, volumeMounts)

        self._extend_for_gcs(env, volumeAttach, volumeMounts)
        self._extend_for_aws(env, volumeAttach, volumeMounts)
        self._extend_for_azure(env, volumeAttach, volumeMounts)

        init_containers = [
            {
                "name": "{{ .Chart.Name }}-deployer",
                "{{- if .Values.deployer.image }}": None,
                "image": "{{ .Values.deployer.image }}",
                "{{ else }}": None,
                '{{ required (printf "Deployer container image is required.") nil }}': None,
                "{{ end }}": None,
                "imagePullPolicy": "{{ .Values.pullPolicy }}",
                "command": ["bash", "-c", "--", str(self.entrypoint)],
                "env": env,
                "resources": {
                    "{{- if .Values.gpu.mig }}": None,
                    "limits{{cond1}}": {
                        "nvidia.com/{{ .Values.gpu.mig }}": "{{ .Values.gpu.limit }}",
                    },
                    "{{ else }}": None,
                    "limits{{cond2}}": {
                        "nvidia.com/gpu": "{{ .Values.gpu.limit }}",
                    },
                    "{{ end }}": None,
                },
                "volumeMounts": volumeMounts,
            }
        ]

        triton_command = TritonServer.command(
            framework=self.framework,
            repository_path=Paths.MODEL_REPOSITORY_PATH,
            strict_mode=False,
            load_mode=TRITON_LOAD_MODE.POLL_ONCE,
        )

        triton_container = {
            "name": "{{ .Chart.Name }}",
            "image": "{{ .Values.server.image }}",
            "imagePullPolicy": "{{ .Values.pullPolicy }}",
            "command": ["/bin/bash", "-c", "--", triton_command],
            "resources": {
                "{{- if .Values.gpu.mig }}": None,
                "limits{{cond1}}": {
                    "nvidia.com/{{ .Values.gpu.mig }}": "{{ .Values.gpu.limit }}",
                },
                "{{ else }}": None,
                "limits{{cond2}}": {
                    "nvidia.com/gpu": "{{ .Values.gpu.limit }}",
                },
                "{{ end }}": None,
            },
            "ports": [
                {"containerPort": 8000, "name": "http"},
                {"containerPort": 8001, "name": "grpc"},
                {"containerPort": 8002, "name": "metrics"},
            ],
            "livenessProbe": {
                "httpGet": {
                    "path": TritonServer.api_method("livenessProbe"),
                    "port": "http",
                }
            },
            "readinessProbe": {
                "initialDelaySeconds": 10,
                "periodSeconds": 5,
                "httpGet": {
                    "path": TritonServer.api_method("readinessProbe"),
                    "port": "http",
                },
            },
            "volumeMounts": volumeMounts,
        }

        containers = [triton_container]

        spec = {
            "replicas": "{{ .Values.replicaCount }}",
            "selector": {
                "matchLabels": {
                    "app": '{{ template "selector.name" . }}',
                    "release": "{{ .Release.Name }}",
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": '{{ template "selector.name" . }}',
                        "release": "{{.Release.Name}}",
                    }
                },
                "spec": {
                    "{{- if .Values.imagePullSecret }}": None,
                    "imagePullSecrets": [
                        {"name": "{{.Values.imagePullSecret }}"},
                    ],
                    "{{ end{{cond1}} }}": None,
                    "initContainers": init_containers,
                    "containers": containers,
                    "volumes": volumeAttach,
                    "restartPolicy": "Always",
                    "{{- if .Values.gpu.product }}": None,
                    "nodeSelector": {
                        "nvidia.com/gpu.product": "{{ .Values.gpu.product }}",
                    },
                    "{{ end{{cond2}} }}": None,
                },
            },
        }

        deployment["metadata"] = metadata
        deployment["spec"] = spec

        return deployment

    def _extend_for_gcs(self, env: typing.List, volumeAttach: typing.List, volumeMounts: typing.List):
        condition = ".Values.deployer.gcsCredentialsFile"
        self._extend_env_for_gcs(env, condition)
        self._extend_volumes_for_gcs(volumeAttach, volumeMounts, condition)

    def _extend_volumes_for_gcs(self, volumeAttach: typing.List, volumeMounts: typing.List, condition: str):
        attach = [
            {
                "name": "gcs-credentials-file",
                "secret": {
                    "secretName": f"{{{{ {condition} }}}}",
                },
            }
        ]
        wrap(lst=attach, condition=condition)
        volumeAttach.extend(attach)

        mount = [
            {
                "mountPath": "/var/credentials/gcs-credentials-file",
                "name": "gcs-credentials-file",
                "readOnly": True,
            }
        ]

        wrap(lst=mount, condition=condition)
        volumeMounts.extend(mount)

    def _extend_env_for_gcs(self, env: typing.List, condition: str):
        env_var = [
            {
                "name": "GOOGLE_APPLICATION_CREDENTIALS",
                "value": "/var/credentials/gcs-credentials-file/credentials.json",
            }
        ]
        wrap(lst=env_var, condition=condition)
        env.extend(env_var)

    def _extend_for_aws(self, env: typing.List, volumeAttach: typing.List, volumeMounts: typing.List):
        condition = ".Values.deployer.awsCredentialsFile"
        self._extend_env_for_aws(env, condition)

    def _extend_env_for_aws(self, env: typing.List, condition: str):
        env_vars = [
            {
                "name": "AWS_ACCESS_KEY_ID",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{{{{ {condition} }}}}",
                        "key": "aws_access_key_id",
                    },
                },
            },
            {
                "name": "AWS_SECRET_ACCESS_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{{{{ {condition} }}}}",
                        "key": "aws_secret_access_key",
                    },
                },
            },
            {
                "name": "AWS_SESSION_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{{{{ {condition} }}}}",
                        "key": "aws_session_token",
                    },
                },
            },
        ]
        wrap(env_vars, condition=condition)
        env.extend(env_vars)

    def _extend_for_azure(self, env: typing.List, volumeAttach: typing.List, volumeMounts: typing.List):
        condition = ".Values.deployer.azureCredentialsFile"
        self._extend_env_for_azure(env, condition)

    def _extend_env_for_azure(self, env: typing.List, condition: str):
        env_vars = [
            {
                "name": "AZURE_STORAGE_CONNECTION_STRING",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{{{{ {condition} }}}}",
                        "key": "azure_storage_connection_string",
                    }
                },
            }
        ]
        wrap(env_vars, condition=condition)
        env.extend(env_vars)
