Feature: Save output package

    Triton Model Navigator save Triton package on output and let user to set the path

    Background:
        Given the TorchScript/simple model with simple config file
        And the target_formats config parameter is set to torchscript

    Scenario: User not set output package
        When I execute optimize command
        Then the command should succeeded
        And the identity_scripted.triton.nav exists

    Scenario: User not set custom output package
        Given the output_package config parameter is set to test.package.nav
        When I execute optimize command
        Then the command should succeeded
        And the test.package.nav exists
