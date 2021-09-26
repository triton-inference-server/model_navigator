Feature: Model Profiling over backend parameters

    The Triton Model Navigator `profile` command let user evaluate a model on the Triton Inference Server
    in order to gather statistics for provided search parameters.

    Triton Model Navigator should be able to swipe over backend parameters.

    Background:
        Given the TorchScript/simple model with simple config file
        And the model_repository config parameter is set to profile/my-model-store

    Scenario: User uses Model Analyzer Configuration Search over Multiple Models
        Given the max_batch_size config parameter is set to 4
        And the model_name config parameter is set to my_model
        And the triton_backend_parameters config parameter is set to param1=a param2=2
        When I execute triton-config-model command
        Then the command should succeeded
        Given the config_search_backend_parameters config parameter is set to param1=a,b,c param3=0.1,0.9
        And removed the max_batch_size config parameter
        And removed the model_name config parameter
        And removed the triton_backend_parameters config parameter
        When I execute profile command
        Then the command should succeeded
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "a"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "a"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.9"}}, "cpu_only": false}
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "b"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "b"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.9"}}, "cpu_only": false}
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "c"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.1"}}, "cpu_only": false}
            {"maxBatchSize": 4, "parameters": {"param1": {"stringValue": "c"}, "param2": {"stringValue": "2"}, "param3": {"stringValue": "0.9"}}, "cpu_only": false}
        And the my_model model was profiled with 1 concurrency levels

