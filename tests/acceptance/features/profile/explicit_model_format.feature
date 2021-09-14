Feature: Model Automatic Profiling Explicit Model Format

    The Triton Model Navigator `profile` command let user evaluate a model on the Triton Inference Server
    in order to gather statistics for provided search parameters and explicit model format is passed.

    Triton Model Navigator should be able to use Triton Model Analyzer Automatic Configuration Search functionality.

    ref: https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md#automatic-configuration-search

    Background:
        Given the TorchScript/simple_no_extension model with simple_no_extension config file
        And the model_repository config parameter is set to profile/my-model-store

    Scenario: User uses Model Analyzer Navigator Convert Model With Explicit Format
        Given the max_batch_size config parameter is set to 4
        And the model_name config parameter is set to my_model
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_max_concurrency config parameter is set to 4
        And the config_search_max_instance_count config parameter is set to 2
        And the config_search_max_preferred_batch_size config parameter is set to 16
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        When I execute profile command
        Then the command should succeeded
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 2, "kind": "KIND_GPU"}], "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 2, "kind": "KIND_GPU"}], "dynamicBatching": {}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [1]}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 2, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [1]}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [2]}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 2, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [2]}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [4]}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 2, "kind": "KIND_GPU"}], "dynamicBatching": {"preferredBatchSize": [4]}, "cpu_only": false}
        And the my_model model was profiled with 1 2 4 concurrency levels
