Feature: Config models on Triton Inference Server with explicit model format

    Scenario: User successfully config ONNX model on Triton Inference Server with explicit model format
        Given the ONNX/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to onnx
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "maxBatchSize": 1, "cpu_only": false}

    Scenario: User successfully config TorchScript model on Triton Inference Server with explicit model format
        Given the TorchScript/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to torchscript
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "pytorch", "maxBatchSize": 1, "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "cpu_only": false}

    Scenario: User successfully config Torch-TRT model on Triton Inference Server with explicit model format
        Given the TorchScript/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to torch-trt
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "pytorch", "maxBatchSize": 1, "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "cpu_only": false}

    Scenario: User successfully config TF-SavedModel model on Triton Inference Server with explicit model format
        Given the TorchScript/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to tf-savedmodel
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "tensorflow", "maxBatchSize": 1, "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "cpu_only": false}

    Scenario: User successfully config TF-TRT model on Triton Inference Server with explicit model format
        Given the TorchScript/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to tf-trt
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "tensorflow", "maxBatchSize": 1,  "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "cpu_only": false}

    Scenario: User successfully config TRT model on Triton Inference Server with explicit model format
        Given the TorchScript/no_extension model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the model_format config parameter is set to trt
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "platform": "tensorrt_plan", "maxBatchSize": 1, "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "optimization": {"cuda": {}}, "cpu_only": false}
