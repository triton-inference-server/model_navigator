<!--
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Helm Charts

## Overview

This section describes how to use the generated [Helm Chart](https://helm.sh/docs/chart_template_guide/getting_started/) to run a model
on Triton Inference Server in a [Kubernetes](https://kubernetes.io/docs/home/) cluster.

**Note** Most of the commands are dependent on the model name passed to Model Navigator on process start.

[Helm](https://helm.sh/) is the package manager for [Kubernetes](https://kubernetes.io/docs/home/) which helps manage
applications in a cluster.

[Helm Charts](https://helm.sh/docs/chart_template_guide/getting_started/) provide an easy way to define how software is
deployed and served on the Kubernetes cluster.

## Serving model on Triton Inference Server

The model Helm Chart provides the definition of steps which are required to be performed in order to serve model inference on
Triton Inference Server in a Kubernetes cluster.

The process consists of two stages:

1. Model preparation for Triton Inference Server
2. Running Triton Inference Server with the model exposed as a Service in a Kubernetes cluster

The first stage is performed on application start and
uses an [Init container](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) concept to download the necessary
files and perform operations required to serve the model on Triton Inference Server.

At application start, the Init container:

1. Downloads the model from the provided URL.
2. Optimizes the model.
3. Prepares the model configuration.
4. Deploys the model to Triton Inference Server model store.
5. Starts Triton Inference Server with the model exposed as a Service.

## Scaling and Load Balancing

The Helm Chart utilizes the Kubernetes concept of a [Service](https://kubernetes.io/docs/concepts/services-networking/service/).
Model; after being deployed to Triton Inference Server and can be accessed by other applications through DNS name.

Kubernetes allows scaling the number of running instances which serves a model through a number of replicas of a given application.

Each newly created instance of an application prepares Triton Inference Server with a model is
accessible under the same Service as Chart's name.

## Quick Start Guide

This section describes the steps which are required to prepare a Docker image and install a Helm Chart on Kubernetes to run
Triton Inference Server with a model in a cluster.

### Generated Helm Charts

Model Navigator is the final step of the process when generating Helm Charts for top N models; based on passed constraints and sorted regards
to selected objectives.

Charts can be found in the charts catalog inside the workspace passed in configuration:
```
{workspace-path}/charts
```

Each generated charts is stored in separate catalogs and contains:
- Charts for model inference
- Charts for inference tester
- toolkit with additional scripts
- Dockerfile which is used in deployment

Example:
```
resnet50
|--- Dockerfile
|--- resnet50-inference
|   |--- Chart.yaml
|   |--- values.yaml
|   |--- templates
|       |--- _helpers.tpl
|       |--- deployment.yaml
|       |--- service.yaml
|
|--- resnet50-tester
|   |--- Chart.yaml
|   |--- values.yaml
|   |--- templates
|       |--- _helpers.tpl
|       |--- deployment.yaml
|
|--- toolkit
   |--- deployer.sh
   |--- tester.sh
```

Every chart generated in the catalog can be built and installed separately.

### Building Docker image for model deployment

This solution provides Dockerfile to prepare Docker images used by the init container and tester chart.

Run the following command from the parent directory, to build the image:

```shell
$ docker build -f Dockerfile -t {image}:{version} .
```

Tag the image with the Docker Registry prefix. For example, if you’re using NGC Docker Registry for your organization, run:

```shell
$ docker tag {image}:{version} nvcr.io/{organization}/{image}:{version}
```

Push the image to a private registry:

```shell
$ docker push nvcr.io/{organization}/{image}:{version}
```

### Install Helm Chart

Helm Chart for top N models are located in the `workspace/charts/{model-variant}` directory.

To install a Helm Chart on your Kubernetes cluster, run:

```shell
$ helm install {INFERENCE_CHART_NAME} {INFERENCE_CHART_NAME} \
 --set deployer.image=nvcr.io/{organization}/{image}:{version} \
 --set deployer.modelUri={url/to/model}
```

The model on Triton Inference Server is deployed in your cluster.

For more information about Helm, install, and other operations, refer to this [documentation](https://helm.sh/docs/intro/using_helm/).

### Optional customizations

Helm Chart for inference allows providing customization on installation time by overriding default values stored in:

```
{INFERENCE_CHART_NAME}/values.yaml
```

For example, in order to provide a different model configuration, you can override the values on installation time:

```shell
$ helm install {INFERENCE_CHART_NAME} {INFERENCE_CHART_NAME} \
 --set deployer.image=nvcr.io/{organization}/{image}:{version} \
 --set deployer.modelUri={url/to/model} \
 --set deployer.maxBatchSize=64
```


#### Adding pull secret for private container registry

If you stored your Docker image in a private container registry, you most likely would need to provide a [secret](https://kubernetes.io/docs/concepts/configuration/secret/)
file name with credentials to the private registry.

During chart installation, you can provide it under `imagePullSecret`:

```
--set imagePullSecret={secret-file-name}
```

#### Scaling number of instances

Helm Chart allows setting up a number of POD instances which are created for the installed chart.

The default value is set to `1`. It means that there will be a single POD with Triton Inference Server which will serve the model.
If you would like to change the number of instances on the chart installation, override the `replicaCount` value:

```
--set replicaCount={Number}
```

#### Setting number of GPU units per Triton Inference Server

During the Helm Chart installation, it’s possible to update the number of GPU units which will be allocated for POD instances.

The default value is set to `1`. It means each created POD instance will acquire 1 GPU unit.
In order to change that number, override the `gpu.limit` value during chart installation:

```
--set gpu.limit={Number}
```
## Downloading Model

Helm Chart provides support for downloading models from the provided URL. The URL must be:
* A model file - `*.pt`, `*.savedmodel`, `*.onnx` or `*.plan` files
* A ZIP archive - `*.zip*` files
* A TAR archive - `*.tar.*` files

In the case of ZIP and TAR archive files, they should contain only one file or catalog in the main archive directory. For example, for TensorFlow SavedModel:
```
ARCHIVE
   |--- model.savedmodel
       |--- model.pb
       |--- variables
```

The archive is automatically unpacked and moved for further optimization.

Helm Chart supports 4 methods for providing files:
- HTTP URL without authorization - `http://` or `https://`,
- AWS S3 URI - `s3://`,
- Azure Cloud Storage URI - `as://`,
- Google Cloud Storage URI - `gs://`,

## Cloud Storages

The default option to provide remote data is to pass uri to a resource which does not require any authorization. In many
cases, you want to access restricted resources saved on a remote storage. For that purpose, Triton Deployer and Helm Chart
support downloading files from:
- [AWS S3 Storage](#aws-s3-storage),
- [Azure Cloud Storage](#azure-cloud-storage),
- [Google Cloud Storage](#google-cloud-storage),

The following sections are dedicated to each provider to provide a deeper understanding on how to use it.

### AWS S3 Storage

Create Kubernetes secret with AWS access key identifier and secret access key (optionally session token, provide empty if not needed):

```shell
$ kubectl create secret generic {SECRET_FILE_NAME} --from-literal=aws_access_key_id={AWS_ACCESS_KEY_ID} --from-literal=aws_secret_access_key={AWS_SECRET_ACCESS_KEY} --from-literal=aws_session_token={AWS_SESSION_TOKEN}
```

For more information, refer to the [AWS Credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

Provide a secret file name on Helm Chart installation:

```
--set deployer.awsCredentialsFile={SECRET_FILE_NAME}
```

Resources uri has to be provided with `s3://` prefix:

```
s3://{bucket}/{file_path}
```

### Azure Cloud Storage

Resources uri has to be provided with `as://` prefix:

```
as://{service_account}/{container}/{file_path}
```

Create Kubernetes secret with Azure Storage Connection string:

```shell
$ kubectl create secret generic {SECRET_FILE_NAME} --from-literal=azure_storage_connection_string={AZURE_STORAGE_CONNECTION_STRING}
```

Provide secret file name on Helm Chart installation:

```
--set deployer.azureCredentialsFile={SECRET_FILE_NAME}
```

### Google Cloud Storage

Create Kubernetes secret with Service Account:

```shell
$ kubectl create secret generic {SECRET_FILE_NAME} --from-file=credentials.json={SERVICE ACCOUNT KEY FILE}
```

Provide secret file name on Helm Chart installation:

```
--set deployer.gcsCredentialsFile={SECRET_FILE_NAME}
```

Resources uri has to be provided with `gs://` prefix:

```
gs://{bucket}/{file_path}
```

## Multi-Instance GPU

### Overview

[Multi-Instance GPU (MIG)](https://developer.nvidia.com/blog/getting-kubernetes-ready-for-the-a100-gpu-with-multi-instance-gpu/)  is a
new feature of the latest generation of NVIDIA GPUs, such as NVIDIA DGX A100. It enables users to maximize the utilization of a
single GPU by running multiple GPU workloads concurrently as if there were multiple smaller GPUs. MIG supports running
multiple workloads in parallel on a single A100 GPU or allowing multiple users to share an A100 GPU with hardware-level
isolation and quality of service.

### Strategies

Helm Chart supports all strategies for MIG in Kubernetes:

- None - MIG is disabled on GPU,
- Single - node expose a single type of MIG device across all its GPUs,
- Mixed - node exposes a mixture of different MIG device types and GPUs in non-MIG mode across all its GPUs.

#### None

The default strategy supported by Helm Chart is `None`. When MIG is disabled on NVIDIA A100 GPU, in order to install Helm Chart
supporting this strategy, run:

```shell
$ helm install {INFERENCE_CHART_NAME} {INFERENCE_CHART_NAME} \
 --set deployer.image=nvcr.io/{organization}/{image}:{version} \
 --set deployer.modelUri={url/to/model}
```

#### Single

In the case of a single strategy, query the MIG partition using the same approach as presented in the Node Selector
section with the correct partition label. For example:

```
--set gpu.product=A100-SXM4-40GB-MIG-1g.5gb
```

#### Mixed

A mixed strategy is necessary to provide GPU product and MIG partition information separately. For example:

```
--set gpu.product=A100-SXM4-40GB \
--set gpu.mig=mig-1g.5gb
```

**Note**: MIG cannot be combined with a number of GPUs - Triton Inference Server will always acquire a single partition on which it will be running.


## Testing Deployed Model

Helm Chart for inference Model Navigator, at the end of the process generates a tester chart to evaluate the deployed models in a cluster.

The chart contains a simple performance test to evaluate the deployed model. Deployment is based on Kubernetes Job which runs a set of performance tests.

It’s necessary to use the same image which has been prepared for inference Helm Chart. In order to deploy tester job, run:
```shell
$ helm install {TESTER_CHART_NAME} {TESTER_CHART_NAME} \
 --set image=nvcr.io/{organization}/{image}:{version}
```
**Note**: You might need to provide `imagePullSecret` if you use a private container registry.

At the end of the job, you will obtain the performance results in the log.

After the job is finished, to remove it, run:

```shell
$ helm uninstall {TESTER_CHART_NAME}
```
