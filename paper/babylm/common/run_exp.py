"""
This runs a number of experiments by calling out to run_single_exp.py

I'm not sure why, but to make running everything easier, we have
- top level run file that iterates over models + tests
- next level run file that runs each of those

Then we have complexity depending on whether we run on cluster or locally from pycharm.
In the former case, because of how I've set up imports, we have to manually set the python path for imports
    -> see fix_path

To use this, follow guidance in
- run_babylm_top
- run_babylm_single

"""
import csv
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
import pprint
from typing import Callable, NamedTuple, Optional

# todo: verify that this will work, since this file gets loaded after run_babylm calls fixpath
#   otherwise need to call fixpath and then import
from babylm.config_babylm import ModelList

class DataRun(NamedTuple):
    model_name: str
    model_idx: int
    test_fn: Callable
    test_idx: int


# Function to execute the run() function in a new Python process
def exec_cmd_for_run(
        datarun: DataRun,
        exec_path: Path | str,
        output_dir: Path,
        test_list: list[Callable],
        model_list: ModelList
):
    """ Executes `run(model_idx, test_idx)` in a fresh Python process. """

    model_short_name = model_list.model_shortname(datarun.model_name)
    fcn_name = test_list[datarun.test_idx].__name__
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path = output_dir/"results.csv"
    log_path = output_dir / "logs" / f"{model_short_name}___{fcn_name}___{date_str}.log"
    print(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(os.path.isdir(os.path.dirname(log_path)))

    # Construct the command to call this script with fresh Python env
    cmd = [
        # conda will nicely be picked up presuming, that it was activated when we are running this file
        sys.executable,
        # "./proj/cxs_are_revealed/exp/exp/run_single_exp.py",
        # "./run_single_exp.py",
        exec_path,
        str(datarun.model_idx),
        str(datarun.test_idx),
        output_path
    ]
    print(f"running cmd:\n\t{cmd}")

    # Open a log file for stdout/stderr redirection
    with open(log_path, "w") as log_file:
        # subprocess.run(cmd, stdout=log_file, stderr=log_file, check=True)
        subprocess.run(cmd, check=True)

def run_all(
        output_dir: Path,
        test_list: list[Callable],
        model_list: ModelList,
        exec_path: Path | str
):
    for test_idx, test in enumerate(test_list):
        for model_idx, model in enumerate(model_list.get_model_iterator()):
            dr = DataRun(
                model, model_idx, test, test_idx
            )
            pprint.pp(dr)
            try:
                exec_cmd_for_run(
                    dr,
                    exec_path,
                    output_dir,
                    test_list,
                    model_list
                )
            except Exception as e:
                print(e)
                traceback.print_exc()

##############
####### single exp code
# will be called by once per model/ test

class OutputRow(NamedTuple):
    model_name: str
    test_name: str
    date: str
    result: Optional[tuple]


def append_to_csv(file_path: str, row: OutputRow):
    """Appends a row to an existing CSV file."""
    row_to_write = [
        row.model_name,
        row.test_name,
        row.date,
    ]
    if row.result:
        if isinstance(row.result, tuple):
            row_to_write.extend(row.result)
        else:
            row_to_write.append(row.result)
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row_to_write)  # Append a single row

def single_run_log_config():
    # note logging will already go to file bc of calling file (run_exp.py)
    logging.basicConfig(level=logging.INFO)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(date_str)
    logging.info(f"running for {pprint.pformat(sys.argv)}")
    logging.info(sys.executable)



# this function will be executed by new python env
def single_run(
        model_idx: int,
        test_idx: int,
        test: Callable,
        model_name: str,
        date_str: str,
        output_csv_path: str
):
    try:
        # todo: make sure this is okay
        d = {"model_idx": model_idx,
             "test_idx": test_idx}
        res = test(**d)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        res = None

    output = OutputRow(
        model_name,
        test.__name__,
        date_str ,
        res
    )
    append_to_csv(output_csv_path, output)
    logging.warning(output)

def single_run_main(
        model_list: ModelList,
        test_list: list
):
    assert len(sys.argv) == 4
    model_idx = int(sys.argv[1])

    model_name, mlm = model_list.get_model_by_idx(model_idx)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    test_idx = int(sys.argv[2])
    test = test_list[test_idx]
    output_csv_path = sys.argv[3]
    logging.info(f"{model_name}, {test}")
    # print("NOW RUN")
    single_run(
        model_idx,
        test_idx,
        test,
        model_name,
        date_str,
        output_csv_path
    )


