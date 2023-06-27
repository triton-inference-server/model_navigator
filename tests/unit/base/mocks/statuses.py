# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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


def status_dict_v0_1_0():
    return {
        "format_version": "0.1.0",
        "uuid": "cd87c5f0-0839-11ed-9198-d45d6452c4ba",
        "git_info": {},
        "environment": {
            "cpu": {
                "name": "CPU",
                "physical_cores": 10,
                "logical_cores": 20,
                "min_frequency": 1600.0,
                "max_frequency": 4000.0,
            },
            "memory": "31.0G",
            "gpu": {
                "name": "Quadro RTX 6000",
                "driver_version": "470.82.00",
                "memory": "24220 MiB",
                "tdp": "260.00 W",
                "cuda_version": "11.4",
            },
            "os": {"name": "posix", "platform": "Linux", "release": "5.11.0-40-generic"},
            "python_version": "3.8.10",
            "python_packages": {
                "numpy": "1.19.4",
            },
            "libraries": {
                "CUDA_DRIVER_VERSION": "510.39.01",
            },
        },
        "export_config": {
            "framework": "tensorflow2",
            "model_name": "navigator_model",
            "target_formats": ["tf-savedmodel", "tf-trt", "onnx", "trt"],
            "sample_count": 100,
            "batch_dim": 0,
            "seed": 0,
            "timestamp": "2022-07-20T14:39:36.545118",
            "from_source": True,
            "max_batch_size": 4,
            "max_workspace_size": 8589934592,
            "target_precisions": ["fp32", "fp16"],
            "trt_dynamic_axes": {"input__0": {"0": [1, 3, 4], "1": [3, 3, 3]}},
            "minimum_segment_size": 3,
            "target_device": "cpu",
            "opset": 14,
            "dynamic_axes": {"input__0": [0]},
            "onnx_runtimes": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        "model_status": [
            {
                "format": "tf-savedmodel",
                "path": "tf-savedmodel/model.savedmodel",
                "runtime_results": [
                    {
                        "runtime": "TensorFlowExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    }
                ],
            },
            {
                "format": "onnx",
                "path": "onnx/model.onnx",
                "runtime_results": [
                    {
                        "runtime": "CUDAExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    },
                    {
                        "runtime": "CPUExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    },
                ],
            },
            {
                "format": "trt",
                "path": "trt-fp32/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    }
                ],
                "precision": "fp32",
            },
            {
                "format": "trt",
                "path": "trt-fp16/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    }
                ],
                "precision": "fp16",
            },
            {
                "format": "tf-trt",
                "path": "tf-trt-fp32/model.savedmodel",
                "runtime_results": [
                    {
                        "runtime": "TensorFlowExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    }
                ],
                "precision": "fp32",
            },
            {
                "format": "tf-trt",
                "path": "tf-trt-fp16/model.savedmodel",
                "runtime_results": [
                    {
                        "runtime": "TensorFlowExecutionProvider",
                        "status": "OK",
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {"batch_size": 1, "latency": 1.0, "throughput": 1500},
                        ],
                        "err_msg": {},
                        "verified": False,
                    }
                ],
                "precision": "fp16",
            },
        ],
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
    }


def status_dict_v0_1_2():
    return {
        "format_version": "0.1.2",
        "model_navigator_version": "0.3.2",
        "uuid": "9f963a60-0838-11ed-b7bd-d45d6452c4ba",
        "git_info": {},
        "environment": {
            "cpu": {
                "name": "CPU",
                "physical_cores": 10,
                "logical_cores": 20,
                "min_frequency": 1600.0,
                "max_frequency": 4000.0,
            },
            "memory": "31.0G",
            "gpu": {
                "name": "NVIDIA GPU",
                "driver_version": "510.39.01",
                "memory": "10240 MiB",
                "tdp": "250.00 W",
                "cuda_version": "11.4",
            },
            "os": {"name": "posix", "platform": "Linux", "release": "5.11.0-40-generic"},
            "python_version": "3.8.12",
            "python_packages": {
                "numpy": "1.19.4",
            },
            "libraries": {
                "CUDA_DRIVER_VERSION": "510.39.01",
            },
        },
        "export_config": {
            "framework": "pytorch",
            "model_name": "navigator_model",
            "target_formats": ["torchscript", "onnx", "torch-trt", "trt"],
            "sample_count": 100,
            "batch_dim": 0,
            "seed": 0,
            "timestamp": "2022-07-20T14:31:20.684175",
            "_input_names": None,
            "_output_names": None,
            "from_source": True,
            "max_batch_size": 4,
            "optimization_profile": {
                "batch_sizes": None,
                "measurement_interval": 5000,
                "measurement_mode": "count_windows",
                "measurement_request_count": 50,
                "stability_percentage": 10.0,
                "max_trials": 10,
            },
            "max_workspace_size": 8589934592,
            "target_precisions": ["fp32", "fp16"],
            "precision_mode": "single",
            "trt_dynamic_axes": {"input__0": {"0": [1, 3, 4], "1": [3, 3, 3]}},
            "minimum_segment_size": None,
            "target_jit_type": ["script", "trace"],
            "target_device": "cuda",
            "opset": 14,
            "dynamic_axes": {"input__0": [0]},
            "onnx_runtimes": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "atol": None,
            "rtol": None,
        },
        "model_status": [
            {
                "format": "torchscript",
                "path": "torchscript-script/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": None,
            },
            {
                "format": "torchscript",
                "path": "torchscript-trace/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": None,
            },
            {
                "format": "onnx",
                "path": "onnx/model.onnx",
                "runtime_results": [
                    {
                        "runtime": "CUDAExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                    {
                        "runtime": "CPUExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                ],
                "torch_jit": None,
                "precision": None,
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-script-fp32/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp32",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-script-fp16/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp16",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-trace-fp32/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp32",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-trace-fp16/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp16",
            },
            {
                "format": "trt",
                "path": "trt-fp32/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": None,
                "precision": "fp32",
            },
            {
                "format": "trt",
                "path": "trt-fp16/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": None,
                "precision": "fp16",
            },
        ],
        "input_metadata": [{"name": "input__0", "shape": [-1, 3], "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": [1, 5], "dtype": "float32"}],
    }


def status_dict_v0_1_3():
    return {
        "format_version": "0.1.3",
        "model_navigator_version": "0.3.3",
        "uuid": "cf9faf36-0994-11ed-a713-d45d6452c4ba",
        "git_info": {},
        "environment": {
            "cpu": {
                "name": "CPU",
                "physical_cores": 10,
                "logical_cores": 20,
                "min_frequency": 1600.0,
                "max_frequency": 4000.0,
            },
            "memory": "31.0G",
            "gpu": {
                "name": "NVIDIA GPU",
                "driver_version": "510.39.01",
                "memory": "10240 MiB",
                "tdp": "250.00 W",
                "cuda_version": "11.4",
            },
            "os": {"name": "posix", "platform": "Linux", "release": "5.11.0-40-generic"},
            "python_version": "3.8.12",
            "python_packages": {
                "numpy": "1.19.4",
            },
            "libraries": {
                "CUDA_DRIVER_VERSION": "510.39.01",
            },
        },
        "export_config": {
            "framework": "pytorch",
            "model_name": "navigator_model",
            "target_formats": ["torchscript", "onnx", "torch-trt", "trt"],
            "sample_count": 100,
            "batch_dim": 0,
            "seed": 0,
            "timestamp": "2022-07-22T08:03:46.136494",
            "_input_names": None,
            "_output_names": None,
            "from_source": True,
            "max_batch_size": 4,
            "optimization_profile": {
                "batch_sizes": None,
                "measurement_interval": 5000,
                "measurement_mode": "count_windows",
                "measurement_request_count": 50,
                "stability_percentage": 10.0,
                "max_trials": 10,
            },
            "max_workspace_size": 8589934592,
            "target_precisions": ["fp32", "fp16"],
            "precision_mode": "single",
            "trt_dynamic_axes": None,
            "minimum_segment_size": None,
            "target_jit_type": ["script", "trace"],
            "target_device": "cuda",
            "opset": 14,
            "dynamic_axes": None,
            "onnx_runtimes": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "atol": None,
            "rtol": None,
        },
        "model_status": [
            {
                "format": "torchscript",
                "path": "torchscript-script/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": None,
            },
            {
                "format": "torchscript",
                "path": "torchscript-trace/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": None,
            },
            {
                "format": "onnx",
                "path": "onnx/model.onnx",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    },
                    {
                        "runtime": "CUDAExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": None,
                        "verified": False,
                    },
                    {
                        "runtime": "CPUExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": None,
                        "verified": False,
                    },
                ],
                "torch_jit": None,
                "precision": None,
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-script-fp32/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp32",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-script-fp16/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp16",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-trace-fp32/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp32",
            },
            {
                "format": "torch-trt",
                "path": "torch-trt-trace-fp16/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp16",
            },
            {
                "format": "trt",
                "path": "trt-fp32/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": None,
                "precision": "fp32",
            },
            {
                "format": "trt",
                "path": "trt-fp16/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": None,
                "precision": "fp16",
            },
        ],
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
        "dataloader_trt_profile": {},
    }


def status_dict_v0_1_4():
    return {
        "format_version": "0.1.4",
        "model_navigator_version": "0.3.8",
        "uuid": "2a51c806-6c32-11ed-baa8-46204d2380e7",
        "git_info": {},
        "environment": {
            "cpu": {
                "name": "CPU",
                "physical_cores": 10,
                "logical_cores": 20,
                "min_frequency": 1600.0,
                "max_frequency": 4000.0,
            },
            "memory": "31.0G",
            "gpu": {
                "name": "NVIDIA GPU",
                "driver_version": "510.39.01",
                "memory": "10240 MiB",
                "tdp": "250.00 W",
                "cuda_version": "11.4",
            },
            "os": {"name": "posix", "platform": "Linux", "release": "5.11.0-40-generic"},
            "python_version": "3.8.12",
            "python_packages": {
                "numpy": "1.19.4",
            },
            "libraries": {
                "CUDA_DRIVER_VERSION": "510.39.01",
            },
        },
        "export_config": {
            "framework": "pytorch",
            "model_name": "navigator_model",
            "target_formats": ["torchscript", "onnx", "torch-trt", "trt"],
            "sample_count": 100,
            "batch_dim": 0,
            "seed": 0,
            "timestamp": "2022-11-24T19:57:03.145190",
            "_input_names": ["INPUT__0"],
            "_output_names": ["OUTPUT__0"],
            "from_source": False,
            "max_batch_size": 256,
            "optimization_profile": {
                "batch_sizes": None,
                "measurement_interval": 5000,
                "measurement_mode": "count_windows",
                "measurement_request_count": 50,
                "stability_percentage": 10.0,
                "max_trials": 10,
            },
            "max_workspace_size": 8589934592,
            "target_precisions": ["fp32", "fp16"],
            "precision_mode": "hierarchy",
            "trt_dynamic_axes": None,
            "minimum_segment_size": None,
            "target_jit_type": ["script", "trace"],
            "target_device": "cuda",
            "opset": 14,
            "dynamic_axes": {"INPUT__0": {"0": "batch"}},
            "runtimes": ["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider"],
            "jit_compile": None,
            "enable_xla": None,
            "atol": None,
            "rtol": None,
            "verbose": True,
            "debug": False,
        },
        "model_status": [
            {
                "format": "torchscript",
                "path": "torchscript-script/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": None,
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "torchscript",
                "path": "torchscript-trace/model.pt",
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": None,
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "onnx",
                "path": "onnx/model.onnx",
                "runtime_results": [
                    {
                        "runtime": "CUDAExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                    {
                        "runtime": "CPUExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    },
                ],
                "torch_jit": None,
                "precision": None,
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "torch-trt",
                "path": None,
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp32",
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "torch-trt",
                "path": None,
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    }
                ],
                "torch_jit": "script",
                "precision": "fp16",
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "torch-trt",
                "path": None,
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp32",
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "torch-trt",
                "path": None,
                "runtime_results": [
                    {
                        "runtime": "PyTorchExecutionProvider",
                        "status": "FAIL",
                        "err_msg": {},
                        "tolerance": None,
                        "performance": None,
                        "verified": False,
                    }
                ],
                "torch_jit": "trace",
                "precision": "fp16",
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "trt",
                "path": "trt-fp32/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                    {
                        "runtime": "TrtexecExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                ],
                "torch_jit": None,
                "precision": "fp32",
                "enable_xla": None,
                "jit_compile": None,
            },
            {
                "format": "trt",
                "path": "trt-fp16/model.plan",
                "runtime_results": [
                    {
                        "runtime": "TensorrtExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                    {
                        "runtime": "TrtexecExecutionProvider",
                        "status": "OK",
                        "err_msg": {},
                        "tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}],
                        "performance": [
                            {
                                "batch_size": 1,
                                "avg_latency": 1.0,
                                "std_latency": 0.0,
                                "p50_latency": 1.0,
                                "p90_latency": 1.0,
                                "p95_latency": 1.0,
                                "p99_latency": 1.0,
                                "throughput": 1500.0,
                                "request_count": 50,
                            }
                        ],
                        "verified": False,
                    },
                ],
                "torch_jit": None,
                "precision": "fp16",
                "enable_xla": None,
                "jit_compile": None,
            },
        ],
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
        "dataloader_trt_profile": {},
    }


def status_dict_v0_2_0():
    return {
        "format_version": "0.2.0",
        "model_navigator_version": "0.4.0",
        "uuid": "1",
        "timestamp": "2022-11-24T19:57:03.145190",
        "environment": {},
        "config": {
            "framework": "torch",
            "target_device": "cpu",
            "runner_names": ["DummySourceTorchRunner", "DummyTorchScriptRunner"],
            "verbose": False,
            "debug": False,
            "target_formats": ["torch", "torchscript"],
            "sample_count": 1,
            "custom_configs": {
                "Onnx": {
                    "opset": 13,
                },
            },
        },
        "models_status": {
            "torch": {
                "model_config": {
                    "format": "torch",
                    "key": "torch",
                    "path": "torch/----",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torch/format.log",
                },
                "runners_status": {
                    "DummySourceTorchRunner": {
                        "runner_name": "DummySourceTorchRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-script": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-script",
                    "path": "torchscript-script/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-script/format.log",
                    "jit_type": "script",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-trace": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-trace",
                    "path": "torchscript-trace/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-trace/format.log",
                    "jit_type": "trace",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "onnx": {
                "model_config": {
                    "format": "onnx",
                    "key": "onnx",
                    "path": "onnx/model.onnx",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "onnx/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "OnnxCUDA",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "trt-fp16": {
                "model_config": {
                    "format": "trt",
                    "key": "trt-fp16",
                    "path": "trt-fp16/model.plan",
                    "parent_path": "onnx/model.onnx",
                    "parent_key": "onnx",
                    "log_path": "trt-fp16/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "TensorRT",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
        },
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
        "dataloader_trt_profile": {},
        "dataloader_max_batch_size": 2,
    }


def status_dict_v0_2_1():
    return {
        "format_version": "0.2.1",
        "model_navigator_version": "0.5.0",
        "uuid": "1",
        "timestamp": "2022-11-24T19:57:03.145190",
        "environment": {},
        "config": {
            "framework": "torch",
            "target_device": "cpu",
            "runner_names": ["DummySourceTorchRunner", "DummyTorchScriptRunner"],
            "verbose": False,
            "debug": False,
            "target_formats": ["torch", "torchscript"],
            "sample_count": 1,
            "custom_configs": {
                "Onnx": {
                    "opset": 13,
                },
            },
            "profiler_config": {
                "run_profiling": False,
                "batch_sizes": [1, 2, 4, 8, 16],
                "measurement_request_count": 50,
                "max_trials": 10,
                "stability_percentage": 10,
                "throughput_cutoff_threshold": 0.05,
            },
        },
        "models_status": {
            "torch": {
                "model_config": {
                    "format": "torch",
                    "key": "torch",
                    "path": "torch/----",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torch/format.log",
                },
                "runners_status": {
                    "DummySourceTorchRunner": {
                        "runner_name": "DummySourceTorchRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-script": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-script",
                    "path": "torchscript-script/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-script/format.log",
                    "jit_type": "script",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-trace": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-trace",
                    "path": "torchscript-trace/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-trace/format.log",
                    "jit_type": "trace",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "onnx": {
                "model_config": {
                    "format": "onnx",
                    "key": "onnx",
                    "path": "onnx/model.onnx",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "onnx/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "OnnxCUDA",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "trt-fp16": {
                "model_config": {
                    "format": "trt",
                    "key": "trt-fp16",
                    "path": "trt-fp16/model.plan",
                    "parent_path": "onnx/model.onnx",
                    "parent_key": "onnx",
                    "log_path": "trt-fp16/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "TensorRT",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
        },
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
        "dataloader_trt_profile": {},
        "dataloader_max_batch_size": 2,
    }


def status_dict_v0_2_2():
    return {
        "format_version": "0.2.2",
        "model_navigator_version": "0.6.0",
        "uuid": "1",
        "timestamp": "2022-11-24T19:57:03.145190",
        "environment": {},
        "config": {
            "framework": "torch",
            "target_device": "cpu",
            "runner_names": ["DummySourceTorchRunner", "DummyTorchScriptRunner"],
            "verbose": False,
            "debug": False,
            "target_formats": ["torch", "torchscript"],
            "sample_count": 1,
            "custom_configs": {
                "Onnx": {
                    "opset": 13,
                },
            },
            "optimization_profile": {
                "batch_sizes": [16],
                "window_size": 50,
                "max_trials": 10,
                "stability_percentage": 10,
                "throughput_cutoff_threshold": 0.05,
            },
        },
        "models_status": {
            "torch": {
                "model_config": {
                    "format": "torch",
                    "key": "torch",
                    "path": "torch/----",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torch/format.log",
                },
                "runners_status": {
                    "DummySourceTorchRunner": {
                        "runner_name": "DummySourceTorchRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-script": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-script",
                    "path": "torchscript-script/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-script/format.log",
                    "jit_type": "script",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "torchscript-trace": {
                "model_config": {
                    "format": "torchscript",
                    "key": "torchscript-trace",
                    "path": "torchscript-trace/model.pt",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "torchscript-trace/format.log",
                    "jit_type": "trace",
                },
                "runners_status": {
                    "DummyTorchScriptRunner": {
                        "runner_name": "DummyTorchScriptRunner",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "onnx": {
                "model_config": {
                    "format": "onnx",
                    "key": "onnx",
                    "path": "onnx/model.onnx",
                    "parent_path": None,
                    "parent_key": None,
                    "log_path": "onnx/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "OnnxCUDA",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
            "trt-fp16": {
                "model_config": {
                    "format": "trt",
                    "key": "trt-fp16",
                    "path": "trt-fp16/model.plan",
                    "parent_path": "onnx/model.onnx",
                    "parent_key": "onnx",
                    "log_path": "trt-fp16/format.log",
                    "jit_type": None,
                },
                "runners_status": {
                    "OnnxCUDA": {
                        "runner_name": "TensorRT",
                        "status": {"Correctness": "OK", "Performance": "OK", "VerifyModel": "OK"},
                        "result": {
                            "Correctness": {
                                "per_output_tolerance": [{"output_name": "output__0", "atol": 0.0, "rtol": 0.0}]
                            },
                            "Performance": {
                                "profiling_results": [
                                    {
                                        "batch_size": 1,
                                        "avg_latency": 1.0,
                                        "std_latency": 0.0,
                                        "p50_latency": 1.0,
                                        "p90_latency": 1.0,
                                        "p95_latency": 1.0,
                                        "p99_latency": 1.0,
                                        "throughput": 1500.0,
                                        "request_count": 50,
                                    }
                                ]
                            },
                        },
                    }
                },
            },
        },
        "input_metadata": [{"name": "input__0", "shape": (-1, 1), "dtype": "float32"}],
        "output_metadata": [{"name": "output__0", "shape": (-1, 1), "dtype": "float32"}],
        "dataloader_trt_profile": {},
        "dataloader_max_batch_size": 2,
    }
