Feature: Set base docker images

    Triton Model Navigator uses Triton Inference Server and Framework docker images.
    Let users enable to set base docker images.

    Because conversion will be run in docker containers, thus convert-model subcommand results
    should have set framework_docker_image parameter.

    Background:
        Given the TorchScript/simple model with simple config file
        And the target_formats config parameter is set to torchscript
#        And the triton_launch_mode config parameter is set to docker

    Scenario: User set just container_version parameter therefore nvcr.io docker images are used
        Given the container_version config parameter is set to 21.05
        When I execute run command
        Then the command should succeeded
        And the convert-model subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.05-py3
        And the triton-config-model subcommand results have succeeded state
        And the profile subcommand results have succeeded state and parameters matching:
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.05-py3
        And the analyze subcommand results have succeeded state
        And the helm-chart-create subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.05-py3
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.05-py3

    Scenario: User set triton_docker_image and framework_docker_image parameter
        Given the container_version config parameter is set to 21.05
        And the framework_docker_image config parameter is set to nvcr.io/nvidia/pytorch:21.02-py3
        And the triton_docker_image config parameter is set to nvcr.io/nvidia/tritonserver:21.02-py3
        When I execute run command
        Then the command should succeeded
        And the convert-model subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.02-py3
        And the triton-config-model subcommand results have succeeded state
        And the profile subcommand results have succeeded state and parameters matching:
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.02-py3
        And the analyze subcommand results have succeeded state
        And the helm-chart-create subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.02-py3
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.02-py3

    Scenario: User set triton_docker_image and config_version parameter
        Given the container_version config parameter is set to 21.05
        And the triton_docker_image config parameter is set to nvcr.io/nvidia/tritonserver:21.02-py3
        When I execute run command
        Then the command should succeeded
        And the convert-model subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.05-py3
        And the triton-config-model subcommand results have succeeded state
        And the profile subcommand results have succeeded state and parameters matching:
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.02-py3
        And the analyze subcommand results have succeeded state
        And the helm-chart-create subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.05-py3
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.02-py3

    Scenario: User set framework_docker_image and config_version parameter
        Given the container_version config parameter is set to 21.05
        And the framework_docker_image config parameter is set to nvcr.io/nvidia/pytorch:21.02-py3
        When I execute run command
        Then the command should succeeded
        And the convert-model subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.02-py3
        And the triton-config-model subcommand results have succeeded state
        And the profile subcommand results have succeeded state and parameters matching:
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.05-py3
        And the analyze subcommand results have succeeded state
        And the helm-chart-create subcommand results have succeeded state and parameters matching:
            framework_docker_image=nvcr.io/nvidia/pytorch:21.02-py3
            triton_docker_image=nvcr.io/nvidia/tritonserver:21.05-py3
