Feature: Config models on Triton Inference Server with batcher

    Background:
        Given the ONNX/simple model with simple config file
        And the model_name config parameter is set to my-model
        And the model_repository config parameter is set to model-store

    Scenario: Max batch size is set passed value when no options passed
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "maxBatchSize": 1, "cpu_only": false}

    Scenario: Max batch size is set passed value when static option used
        Given the batching config parameter is set to static
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "maxBatchSize": 1, "cpu_only": false}

    Scenario: Max batch size is set to 0 when disabled option used
        Given the batching config parameter is set to disabled
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "cpu_only": false}

    Scenario: Dynamic batching is configured when dynamic options is used
         Given the batching config parameter is set to dynamic
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "maxBatchSize": 1, "cpu_only": false, "dynamicBatching": {}}

    Scenario: Dynamic batching is configured when dynamic options is used and additional parameters passed
        Given the batching config parameter is set to dynamic
        And the preferred_batch_sizes config parameter is set to 1 2
        And the max_queue_delay_us config parameter is set to 1000
        When I execute triton-config-model command
        Then the command should succeeded
        And the my-model model config in model-store is equal to
            {"name": "my-model", "backend": "onnxruntime", "maxBatchSize": 1, "cpu_only": false, "dynamicBatching": {"preferredBatchSize": [1, 2], "maxQueueDelayMicroseconds": "1000"}}
