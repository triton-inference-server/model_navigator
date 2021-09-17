from typing import Dict, List


def equal_model_configs_sets(a: List[Dict], b: List[Dict]):
    a = a.copy()
    b = b.copy()

    same = None
    for item in a:
        if item not in b:
            same = False
            break
        b.remove(item)
    if same is None:
        same = not b
    else:
        same = False
    return same
