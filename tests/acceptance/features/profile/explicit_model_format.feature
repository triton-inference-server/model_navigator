Feature: Model Automatic Profiling Explicit Model Format

    The Triton Model Navigator `profile` command let user evaluate a model on the Triton Inference Server
    in order to gather statistics for provided search parameters and explicit model format is passed.

    Triton Model Navigator should be able to use Triton Model Analyzer Automatic Configuration Search functionality.

    ref: https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md#automatic-configuration-search

    Background:
        Given the TorchScript/no_extension model with simple config file
        And the model_repository config parameter is set to profile/my-model-store

    Scenario: User uses Model Analyzer Navigator Model With Explicit Format
        Given the max_batch_size config parameter is set to 2
        And the model_name config parameter is set to my_model
        And the model_format config parameter is set to torchscript
        When I execute triton-config-model command
        Then the command should succeeded
        Given removed the max_batch_size config parameter
        And removed the model_name config parameter
        And removed the model_format config parameter
        When I execute profile command
        Then the command should succeeded
        And the my_model model configs in latest profile checkpoint are
            {"maxBatchSize": 2, "cpu_only": false}
            {"maxBatchSize": 2, "instanceGroup": [{"count": 1, "kind": "KIND_GPU"}], "dynamicBatching": {}, "cpu_only": false}
        And the my_model model was profiled with 1 concurrency levels
