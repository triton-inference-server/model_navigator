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
from .. import helm, internals


class Deployment(helm.Deployment):
    @property
    def volumes(self):
        return [
            helm.Volume(name="shared-dir", env="SHARED_DIR", path=internals.Paths.SHARED_DIR, empty_dir={}),
        ]

    def data(self):
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
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

        env = list()
        volumeMounts = list()
        volumeAttach = list()

        self._prepare_environment(env)
        self._prepare_volumes(env, volumeAttach, volumeMounts)

        container = {
            "name": f"{self.name}",
            "{{- if .Values.image }}": None,
            "image": "{{ .Values.image }}",
            "{{ else }}": None,
            '{{ required (printf "Container image is required.") nil }}': None,
            "{{ end }}": None,
            "imagePullPolicy": "{{ .Values.pullPolicy }}",
            "command": ["bash", "-c", "--", str(self.entrypoint)],
            "env": env,
            "volumeMounts": volumeMounts,
        }

        spec = {
            "template": {
                "spec": {
                    "{{- if .Values.imagePullSecret }}": None,
                    "imagePullSecrets": [
                        {"name": "{{.Values.imagePullSecret }}"},
                    ],
                    "{{ end{{cond1}} }}": None,
                    "containers": [container],
                    "volumes": volumeAttach,
                    "restartPolicy": "OnFailure",
                }
            }
        }

        job["metadata"] = metadata
        job["spec"] = spec

        return job
