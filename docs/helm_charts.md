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

## The `helm-chart-create` Command

The `helm-chart-create` command generates the Helm Chart per selected model configuration.

Using CLI arguments:

```shell
$ model-navigator helm-chart-create --model-name add_sub \
    --model-path model_navigator/examples/quick-start/model.pt \
    --charts-repository navigator_workspace/charts \
    --chart-name add_sub_i0
```

Using YAML file:

```yaml
model_name: add_sub
model_path: model_navigator/examples/quick-start/model.pt
charts_repository: navigator_workspace/charts
chart_name: add_sub_i0
```

Running command using YAML configuration:

```shell
$ model-navigator helm-chart-create --config-path model_navigator.yaml
```

## CLI and YAML Config Options

[comment]: <> (START_CONFIG_LIST)
```yaml
# Name of the model.
model_name: str

# Path to the model file.
model_path: path

# Path to Helm Charts repository.
charts_repository: path

# Name of the chart in Helm Charts repository.
chart_name: str

# Path to the configuration file containing default parameter values to use.
# For more information about configuration files, refer to:
# https://github.com/triton-inference-server/model_navigator/blob/main/docs/run.md
[ config_path: path ]

# Path to the output workspace directory.
[ workspace_path: path | default: navigator_workspace ]

# Clean workspace directory before command execution.
[ override_workspace: boolean ]

# NVIDIA framework and Triton container version to use. For details refer to
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html and
# https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html for details).
[ container_version: str | default: 21.10 ]

# Custom framework docker image to use. If not provided
# nvcr.io/nvidia/<framework>:<container_version>-<framework_and_python_version> will be used
[ framework_docker_image: str ]

# Custom Triton Inference Server docker image to use.
# If not provided nvcr.io/nvidia/tritonserver:<container_version>-py3 will be used
[ triton_docker_image: str ]

# List of GPU UUIDs or Device IDs to be used for the conversion and/or profiling.
# All values have to be provided in the same format.
# Use 'all' to profile all the GPUs visible by CUDA.
[ gpus: str | default: ['all'] ]

# Provide verbose logs.
[ verbose: boolean ]

# Format of the model. Should be provided in case it is not possible to obtain format from model filename.
[ model_format: choice(torchscript, tf-savedmodel, tf-trt, onnx, trt) ]

# Version of model used by the Triton Inference Server.
[ model_version: str | default: 1 ]

# Version of the chart in Helm Charts repository.
[ chart_version: str ]

# Signature of the model inputs.
[ inputs: list[str] ]

# Signature of the model outputs.
[ outputs: list[str] ]

# Target format to generate.
[ target_formats: list[str] ]

# Generate an ONNX graph that uses only ops available in a given opset.
[ onnx_opsets: list[integer] ]

# Configure TensorRT builder for precision layer selection.
[ tensorrt_precisions: list[choice(int8, fp16, fp32, tf32)] | default: ['fp16', 'tf32'] ]

# Select how target precision should be applied during conversion:
# 'hierarchy': enable possible precisions for values passed in target precisions int8 enable tf32, fp16 and int8
# 'single': each precision passed in target precisions is applied separately
# 'mixed': combine both strategies
[ tensorrt_precisions_mode: choice(hierarchy, single, mixed) | default: hierarchy ]

# Enable explicit precision for TensorRT builder when model already contain quantized layers.
[ tensorrt_explicit_precision: boolean ]

# Enable strict types in TensorRT, forcing it to choose tactics based on the layer precision set, even if another
# precision is faster.
[ tensorrt_strict_types: boolean ]

# Enable optimizations for sparse weights in TensorRT.
[ tensorrt_sparse_weights: boolean ]

# The maximum GPU memory in bytes the model can use temporarily during execution for TensorRT acceleration.
[ tensorrt_max_workspace_size: integer ]

# Absolute tolerance parameter for output comparison.
# To specify per-output tolerances, use the format: --atol [<out_name>=]<atol>.
# Example: --atol 1e-5 out0=1e-4 out1=1e-3
[ atol: list[str] | default: ['1e-05'] ]

# Relative tolerance parameter for output comparison.
# To specify per-output tolerances, use the format: --rtol [<out_name>=]<rtol>.
# Example: --rtol 1e-5 out0=1e-4 out1=1e-3
[ rtol: list[str] | default: ['1e-05'] ]

# Maximum batch size allowed for inference.
# A max_batch_size value of 0 indicates that batching is not allowed for the model
[ max_batch_size: integer | default: 32 ]

# Map of features names and minimum shapes visible in the dataset.
# Format: --min-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ min_shapes: list[str] ]

# Map of features names and optimal shapes visible in the dataset.
# Used during the definition of the TensorRT optimization profile.
# Format: --opt-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ opt_shapes: list[str] ]

# Map of features names and maximal shapes visible in the dataset.
# Format: --max-shapes <input0>=D0,D1,..,DN .. <inputN>=D0,D1,..,DN
[ max_shapes: list[str] ]

# Map of features names and range of values visible in the dataset.
# Format: --value-ranges <input0>=<lower_bound>,<upper_bound> ..
# <inputN>=<lower_bound>,<upper_bound> <default_lower_bound>,<default_upper_bound>
[ value_ranges: list[str] ]

# Map of features names and numpy dtypes visible in the dataset.
# Format: --dtypes <input0>=<dtype> <input1>=<dtype> <default_dtype>
[ dtypes: list[str] ]

# Select Backend Accelerator used to serve the model.
[ backend_accelerator: choice(none, amp, trt) ]

# Target model precision for TensorRT acceleration.
[ tensorrt_precision: choice(int8, fp16, fp32) ]

# Enable CUDA capture graph feature on the TensorRT backend.
[ tensorrt_capture_cuda_graph: boolean ]

# Batch sizes that the dynamic batcher should attempt to create.
# In case --max-queue-delay-us is set and this parameter is not, default value will be --max-batch-size.
[ preferred_batch_sizes: list[integer] ]

# Max delay time that the dynamic batcher will wait to form a batch.
[ max_queue_delay_us: integer ]

# Mapping of device kind to model instances count on a single device. Available devices: [cpu|gpu].
# Format: --engine-count-per-device <kind>=<count>
[ engine_count_per_device: list[str] ]

```
[comment]: <> (END_CONFIG_LIST)

## Using Generated Helm Charts

This section describes how to use the generated [Helm Chart](https://helm.sh/docs/chart_template_guide/getting_started/) to run a model
on the Triton Inference Server in a [Kubernetes](https://kubernetes.io/docs/home/) cluster.

**Note** Most of the commands are dependent on the model name passed to the Triton Model Navigator on process start.

[Helm](https://helm.sh/) is the package manager for [Kubernetes](https://kubernetes.io/docs/home/) that helps manage
applications in a cluster.

[Helm Charts](https://helm.sh/docs/chart_template_guide/getting_started/) provide an easy way to define how software is
deployed and served on the Kubernetes cluster.

## Serving Model on the Triton Inference Server

The model Helm Chart provides the definition of steps required to be performed in
order to serve model inference on the Triton Inference Server in a Kubernetes cluster.

The process consists of two stages:

1. Preparing the model for the Triton Inference Server
2. Running the Triton Inference Server with the model exposed as a Service in a Kubernetes cluster

The first stage is performed on application start and
uses an [Init container](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) concept to download the necessary
files and perform operations required to serve the model on Triton Inference Server.

At application start, the Init container:

1. Downloads the model from the provided URL.
2. Optimizes the model.
3. Prepares the model configuration.
4. Deploys the model to the Triton Inference Server model store.
5. Starts Triton Inference Server with the model exposed as a Service.

## Scaling and Load Balancing

The Helm Chart utilizes the Kubernetes concept of a [Service](https://kubernetes.io/docs/concepts/services-networking/service/).
Model after being deployed to Triton Inference Server can be accessed by other applications through DNS name.

Kubernetes allows scaling the number of running instances that serves a model through a number of replicas of a given application.

Each newly created instance of an application prepares Triton Inference Server with a model that is
accessible under the same Service as Chart's name.

## Quick Start Guide

This section describes the steps that are required to prepare a Docker image and install a Helm Chart on Kubernetes to run
Triton Inference Server with a model in a cluster.

### Generated Helm Charts

The Triton Model Navigator is the final step of the process when generating Helm Charts for top N models based on passed
constraints and sorted in regards to selected objectives.

Charts can be found in the charts catalog inside the workspace passed in configuration:
```
{workspace-path}/charts
```

Each generated charts is stored in separate catalogs and contains:
- Charts for model inference
- Charts for inference tester
- toolkit with additional scripts
- Dockerfile that is used in deployment

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

### Building Docker Image for Model Deployment

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

### Installing the Helm Chart

The Helm Chart for top N models is located in the `workspace/charts/{model-variant}` directory.

To install a Helm Chart on your Kubernetes cluster, run:

```shell
$ helm install {INFERENCE_CHART_NAME} {INFERENCE_CHART_NAME} \
 --set deployer.image=nvcr.io/{organization}/{image}:{version} \
 --set deployer.modelUri={url/to/model}
```

The model on the Triton Inference Server is deployed in your cluster.

For more information about Helm, installation, and other operations, refer to this [documentation](https://helm.sh/docs/intro/using_helm/).

### Optional Customizations

The Helm Chart for inference allows providing customization on installation time by overriding default values stored in:

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


#### Adding Pull Secret for Private Container Registry

If you stored your Docker image in a private container registry, you may need to provide a
a [secret](https://kubernetes.io/docs/concepts/configuration/secret/)
file name with credentials to the private registry.

During chart installation, you can provide it under `imagePullSecret`:

```
--set imagePullSecret={secret-file-name}
```

#### Scaling Number of Instances

The Helm Chart allows setting up a number of POD instances that are created for the installed chart.

The default value is set to `1`, which means there will be a single POD with Triton Inference Server
that serves the model. If you would like to change the number of instances on the chart installation,
override the `replicaCount` value:
```
--set replicaCount={Number}
```

#### Setting number of GPU Units per Triton Inference Server

During the Helm Chart installation, it’s possible to update the number of GPU units that will
be allocated for POD instances.

The default value is set to `1` which means each created POD instance will acquire 1 GPU unit.
In order to change that number, override the `gpu.limit` value during chart installation:

```
--set gpu.limit={Number}
```
## Downloading Model

The Helm Chart provides support for downloading models from the provided URL. The URL must be:
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

Helm Chart supports four methods for providing files:
- HTTP URL without authorization - `http://` or `https://`
- AWS S3 URI - `s3://`
- Azure Cloud Storage URI - `as://`
- Google Cloud Storage URI - `gs://`

## Cloud Storages

The default option to provide remote data is to pass uri to a resource that does not require any authorization. In many
cases, you want to access restricted resources saved on a remote storage. For that purpose, Triton Deployer and Helm Chart
support downloading files from:
- [AWS S3 Storage](#aws-s3-storage)
- [Azure Cloud Storage](#azure-cloud-storage)
- [Google Cloud Storage](#google-cloud-storage)

The following sections are dedicated to each provider to provide a
deeper understanding of how to use their cloud storage.

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
new feature of the latest generation of NVIDIA GPUs, such as the NVIDIA DGX A100. It enables users to maximize the utilization of a
single GPU by running multiple GPU workloads concurrently as if there were multiple smaller GPUs. MIG supports running
multiple workloads in parallel on a single A100 GPU or allowing multiple users to share an A100 GPU with hardware-level
isolation and quality of service.

### Strategies

Helm Chart supports all strategies for MIG in Kubernetes:

- None - MIG is disabled on GPU
- Single - node expose a single type of MIG device across all its GPUs
- Mixed - node exposes a mixture of different MIG device types and GPUs in non-MIG mode across all its GPUs

#### None

The default strategy supported by the Helm Chart is `None`. When MIG is disabled on NVIDIA A100 GPU, in order to install Helm Chart
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

**Note**: MIG cannot be combined with a number of GPUs. The Triton Inference Server will always acquire a single partition on which it will be running.


## Testing Deployed Model

The Helm Chart for inference the Triton Model Navigator, at the end of the process generates a tester chart to evaluate the deployed models in a cluster.

The chart contains a simple performance test to evaluate the deployed model. Deployment is based on Kubernetes Job which runs a set of performance tests.

It’s necessary to use the same image that was prepared for Inference Helm Chart. In order to deploy the tester job, run:
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
