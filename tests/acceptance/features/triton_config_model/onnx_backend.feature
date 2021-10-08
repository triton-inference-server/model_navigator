Feature: Config ONNX models on Triton Inference Server

    Background:
        Given the ONNX/simple model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store

    Scenario: User successfully config ONNX model on Triton Inference Server
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnx", "maxBatchSize": 1, "cpu_only": false}


    Scenario: User successfully config ONNX model with TensorRT acceleration
        Given the backend_accelerator config parameter is set to trt
        And the tensorrt_precision config parameter is set to fp16
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnx", "maxBatchSize": 1, "optimization": {"executionAccelerators": {"gpuExecutionAccelerator": [{"name": "tensorrt", "parameters": {"precision_mode": "FP16", "max_workspace_size_bytes": "4294967296"}}]}}, "cpu_only": false}


    Scenario: User successfully config ONNX model with TensorRT acceleration and custom max workspace size
        Given the backend_accelerator config parameter is set to trt
        And the tensorrt_precision config parameter is set to fp32
        And the tensorrt_max_workspace_size config parameter is set to 1024
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnx", "maxBatchSize": 1, "optimization": {"executionAccelerators": {"gpuExecutionAccelerator": [{"name": "tensorrt", "parameters": {"precision_mode": "FP32", "max_workspace_size_bytes": "1024"}}]}}, "cpu_only": false}


    Scenario: User is informed on missing required parameters while requesting TensorRT acceleration
        Given the backend_accelerator config parameter is set to trt
        When I execute triton-config-model command
        Then the command should failed
        And the --tensorrt-precision is required pattern is present on command output
