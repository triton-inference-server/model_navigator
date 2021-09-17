Feature: Config models on Triton Inference Server with explicit model format

    Background:
        Given the TorchScript/simple_no_extension model with simple_no_extension config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store

    Scenario: User successfully config TorchScript model on Triton Inference Server with explicit model format
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "pytorch", "maxBatchSize": 1, "input": [{"name": "input__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "output": [{"name": "output__0", "dataType": "TYPE_FP32", "dims": ["3", "-1", "-1"]}], "cpu_only": false}
