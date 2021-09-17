import json

from model_analyzer.state.analyzer_state import AnalyzerState

from model_navigator.results import ResultsStore


def get_profile_results(workspace):
    results_store = ResultsStore(workspace)
    command_results = results_store.load("profile")
    # for profile there is single result object
    checkpoint_path = command_results[0].profiling_results_path
    with checkpoint_path.open("r") as checkpoint_file:
        state = AnalyzerState.from_dict(json.load(checkpoint_file))
    profiling_results = state.get("ResultManager.results")
    return profiling_results
