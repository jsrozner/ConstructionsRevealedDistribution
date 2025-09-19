import logging
import os
import traceback
from pathlib import Path
from typing import List, Callable, Optional, Literal


# todo: move this into rozlib utils

def check_dir_exists(
        dir_to_check: Path | str,
        create_ok: bool = False,
        err_msg: Optional[str] = None):
    # check directories
    if os.path.isdir(dir_to_check):
        return

    if not create_ok:
        if err_msg:
            raise Exception(err_msg)
        raise Exception(f"{dir_to_check} is not a directory")

    logging.info(f"dir {dir_to_check} did not exist; creating")
    os.makedirs(dir_to_check, exist_ok=True)

def run_for_file(
        input_file: Path,
        jsonl_output_dir: Path,
        run_fcn: Callable,
        keep_existing = False
):
    # main for loop
    file_id = input_file.name.split(".")[0]
    logging.info(f"Looking at file_id {file_id} ({input_file})")

    check_dir_exists(jsonl_output_dir, create_ok=True)
    tmp_out_path = jsonl_output_dir / f"_tmp_{file_id}.jsonl"
    final_out_path = jsonl_output_dir / f"{file_id}.jsonl"

    if os.path.exists(final_out_path):
        logging.warning(f"Output file {final_out_path} already exists. Skipping")
        return

    # deal with file "locks" / possible failures
    if os.path.exists(tmp_out_path):
        # todo consider allowing parallelism / resumption
        logging.warning(f"Temp path exists already. This means something broke before"
                        f"(or in progress if parallelism is allowed)")
        if keep_existing:
            logging.warning("Keep is set; will not delete")
        else:
            logging.warning("will remove and restart that file")
            os.remove(tmp_out_path)

    #otherwise, create file at tmp path (this process has a "lock")
    tmp_out_path.touch()
    logging.info(f"touched temp path {tmp_out_path}")

    try:
        run_fcn(
            input_file,
            tmp_out_path,
        )
        os.rename(tmp_out_path, final_out_path)
        logging.info(f"Finished processing {final_out_path}")
    except Exception as e:
        logging.error(f"Error processing {file_id} ({input_file}): \n{e}")
        traceback.print_exc()
    # finally:
        # if os.path.exists(tmp_out_path):
        #     os.remove(tmp_out_path)
        #     logging.info(f"Successfully removed temp path {tmp_out_path}")


def run_for_sub_dir(
        input_dir: Path,
        output_dir: Path,
        folder_list: List[str],
        run_fcn: Callable[[Path, Path], None],
        file_limit_ct: Optional[int] = None,
):
    """
    Will parse all files in input_dir/ [folder_list] .txt
    """
    # todo: support multiple folders (eg for glob)
    if len(folder_list) > 1:
        raise NotImplementedError("expects only a single sub dir for run")

    # we keep the same subdir structure that BNC had, with, e.g., bnc/A/A0A.txt
    text_file_input_dir = input_dir/ folder_list[0]
    jsonl_output_dir = output_dir / folder_list[0]

    check_dir_exists(
        text_file_input_dir,
        create_ok=False,
        err_msg= f"{text_file_input_dir} is not a directory"
    )
    check_dir_exists(
        jsonl_output_dir,
        create_ok=True,
        err_msg=f"{text_file_input_dir} is not a directory"
    )

    # run for each file
    file_ct = 0
    for input_file in text_file_input_dir.glob("*.txt"):
        # file processing limit for processing a reduced set
        if file_limit_ct is not None and file_ct >= file_limit_ct:
            break
        file_ct += 1
        run_for_file(input_file, jsonl_output_dir, run_fcn)

