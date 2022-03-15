Feature: Model Profiling of Model Repositories with Multiple Models

    The Triton Model Navigator `profile` command let user evaluate a model on the Triton Inference Server
    in order to gather statistics for provided search parameters.

    Triton Model Navigator should be able to use profile multiple models in single profile run.

    Background:
        Given the TorchScript/simple model with simple config file
        And the model_repository config parameter is set to profile/my-model-store

    Scenario: User uses Model Analyzer Configuration Search over Multiple Models
        Given the max_batch_size config parameter is set to 2
        And the model_name config parameter is set to my_model1
        When I execute triton-config-model command
        Then the command should succeeded
        Given the model_name config parameter is set to my_model2
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_max_instance_count config parameter is set to 1
        And the config_search_max_concurrency config parameter is set to 1
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        When I execute profile command
        Then the command should succeeded
        And the 'run_config_search_disable': False pattern is present on command output
        And the my_model1 model configs in latest profile checkpoint are
            {"maxBatchSize": 2, "cpu_only": false}
            {"maxBatchSize": 2, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {}, "cpu_only": false}
        And the my_model1 model was profiled with 1 concurrency levels
        And the my_model2 model configs in latest profile checkpoint are
            {"maxBatchSize": 2, "cpu_only": false}
            {"maxBatchSize": 2, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {}, "cpu_only": false}
        And the my_model2 model was profiled with 1 concurrency levels

