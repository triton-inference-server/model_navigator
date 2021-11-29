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
from model_navigator.kubernetes.helm import Service as ServiceConfig


class Service(ServiceConfig):
    def data(self):
        service = {
            "apiVersion": "v1",
            "kind": "Service",
        }

        metadata = {
            "name": '{{ template "selector.fullname" . }}',
            "namespace": "{{ .Release.Namespace }}",
            "labels": {
                "app": '{{ template "selector.name" . }}',
                "chart": '{{template "selector.chart" . }}',
                "release": "{{ .Release.Name }}",
                "heritage": "{{ .Release.Service }}",
            },
        }

        spec = {
            "type": "{{ .Values.service.type }}",
            "ports": [
                {"port": 8000, "targetPort": 8000, "name": "http"},
                {"port": 8001, "targetPort": 8001, "name": "grpc"},
                {"port": 8002, "targetPort": 8002, "name": "metrics"},
            ],
            "selector": {
                "app": '{{ template "selector.name" . }}',
                "release": "{{ .Release.Name }}",
            },
        }

        service["metadata"] = metadata
        service["spec"] = spec

        return service
