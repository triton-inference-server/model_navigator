Feature: Error codes

    Scenario: User is informed on missing required parameters with proper error code and message
        Given the ONNX/simple model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store
        And the backend_accelerator config parameter is set to trt
        When I execute triton-config-model command
        Then the command should failed
        And the --tensorrt-precision is required substring is present on command output
