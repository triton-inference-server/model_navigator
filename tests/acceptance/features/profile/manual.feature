Feature: Model Manual Profiling

    The Triton Model Navigator `profile` command let user evaluate a model on the Triton Inference Server
    in order to gather statistics for provided search parameters.

    Triton Model Navigator should be able to use Triton Model Analyzer Manual Configuration Search functionality.

    ref: https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md#manual-configuration-search

    Background:
        Given the TorchScript/simple model with simple config file
        And the model_repository config parameter is set to profile/my-model-store

    Scenario: User uses Model Analyzer Manual Configuration Search
        Given the max_batch_size config parameter is set to 4
        And the model_name config parameter is set to my_model
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_preferred_batch_sizes config parameter is set to 2,4 4
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        When I execute profile command
        Then the command should succeeded
        And the 'run_config_search_disable': True pattern is present on command output
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 4, "cpu_only": false}
            {"maxBatchSize": 4, "dynamicBatching": {"preferredBatchSize": [2, 4]}, "cpu_only": false}
            {"maxBatchSize": 4, "dynamicBatching": {"preferredBatchSize": [4]}, "cpu_only": false}
        And the my_model model was profiled with 1 concurrency levels

    Scenario: User uses Model Analyzer Manual Configuration Search with concurrency levels set
        Given the max_batch_size config parameter is set to 4
        And the model_name config parameter is set to my_model
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_concurrency config parameter is set to 1 2
        And the config_search_max_instance_count config parameter is set to 1
        And the config_search_preferred_batch_sizes config parameter is set to 2,4 4
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        When I execute profile command
        Then the command should succeeded
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 4, "cpu_only": false}
            {"maxBatchSize": 4, "dynamicBatching": {"preferredBatchSize": [2, 4]}, "cpu_only": false}
            {"maxBatchSize": 4, "dynamicBatching": {"preferredBatchSize": [4]}, "cpu_only": false}

        And the my_model model was profiled with 1 2 concurrency levels

    Scenario: User uses Model Analyzer Manual Configuration Search with search over all possible axes
        Given the max_batch_size config parameter is set to 4
        And the model_name config parameter is set to my_model
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_instance_counts config parameter is set to gpu=1,3 cpu=3
        Given the config_search_max_batch_sizes config parameter is set to 4 32
        Given the config_search_preferred_batch_sizes config parameter is set to 2,4 8
        Given the config_search_backend_parameters config parameter is set to param1=value1.1,value1.2 param2=value2.1
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        When I execute profile command
        Then the command should succeeded
        And the 'run_config_search_disable': True pattern is present on command output
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 4, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param2": {"stringValue": "value2.1"}, "param1": {"stringValue": "value1.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.2"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.1"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.2"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.1"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.2"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.1"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [2, 4]}, "parameters": {"param1": {"stringValue": "value1.2"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [8]}, "parameters": {"param2": {"stringValue": "value2.1"}, "param1": {"stringValue": "value1.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [8]}, "parameters": {"param2": {"stringValue": "value2.1"}, "param1": {"stringValue": "value1.2"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [8]}, "parameters": {"param1": {"stringValue": "value1.1"}, "param2": {"stringValue": "value2.1"}}, "cpu_only": false}
            {"maxBatchSize": 32, "instanceGroup": [{"count": 3, "kind": "KIND_GPU"}, {"count": 3, "kind": "KIND_CPU"}], "dynamicBatching": {"preferredBatchSize": [8]}, "parameters": {"param2": {"stringValue": "value2.1"}, "param1": {"stringValue": "value1.2"}}, "cpu_only": false}
        And the my_model model was profiled with 1 concurrency levels
