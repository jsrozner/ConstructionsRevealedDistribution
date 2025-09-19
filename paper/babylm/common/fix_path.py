import os
import pprint
import sys
from pathlib import Path

def add_paths(
        path_list: list[str]
):
    # Add it to sys.path if not already present
    for p in path_list:
        if p not in sys.path:
            print(f"add {p} to path")
            sys.path.insert(0, p)  # Insert at the beginning
    pprint.pp(sys.path)


def fix_path() -> Path:
    # detect if we're on cluster or local
    if not os.getcwd().startswith("/Users/jsrozner"):
        # on cluster
        print("fix_path: WARN: treating as if on cluster (MAIN)")
        print(os.getcwd())
        extra_paths = [
            os.path.abspath("./proj/cxs_are_revealed/src"),
            os.path.abspath("./rozlib-python"),
            # todo: not sure if this is needed
            os.getcwd()
        ]

        add_paths(
            extra_paths
        )

        # todo: check
        # exec_path = os.path.abspath("./proj/cxs_are_revealed/exp/exp/run_single_exp.py")
        # exec_path = os.path.abspath("./proj/cxs_are_revealed/exp/exp/")
        # exec path should be in the same location
        exec_path_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "run_babylm_single.py"
    else:
        # local
        print("fix_path: run local")
        exec_path_dir = Path(os.path.abspath("run_babylm_single.py"))

    return exec_path_dir
