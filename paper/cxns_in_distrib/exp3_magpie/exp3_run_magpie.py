import logging
import os
import traceback
from pathlib import Path
from typing import Optional, List

import torch
from tqdm import tqdm

from lib.common.mlm_singleton import get_singleton_scorer
from lib.scoring_fns import hhi, min_surprisal, ScoreFnTensor
from lib.exp_common.config import default_config
from lib.scoring_fns import surprisal
from lib.utils.utils_run import run_for_sub_dir
from rozlib.libs.common.config_wrapper import set_config
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl, read_from_jsonl
from rozlib.libs.common.logging.logging import setup_logging
from proj.cxs_are_revealed.paper.cxns_in_distrib.exp3_magpie import MagpieEntryForProcessing, MLMResultForSentenceExp6, EntryForProcessing
from rozlib.libs.library_ext_utils.utils_torch import get_device

device = get_device()
mlm = get_singleton_scorer('roberta-large')


def score_exp6(
        logit_outputs: torch.Tensor,
        orig_token_ids: torch.Tensor,
        score_fns: List[ScoreFnTensor]
) -> torch.Tensor:
    """
    # todo: note mostly copied
    Note: this function differs in that we changed the shape of the logits matrix; otherwise the same

    Compute scores for each row in logit_outputs using the provided score functions.

    Args:
    logit_outputs (torch.Tensor): A tensor of shape (sent_len / batch_len, vocab_len) containing logit outputs from the model.
    score_fns (List[Callable]): A list of scoring functions, each taking a row (tensor) as input and returning a float.

    Returns:
    torch.Tensor: A tensor of shape (len(score_fns), len(sent)) containing the scores.
    """
    num_score_fns = len(score_fns)

    # Initialize an empty tensor to hold the results, allocated on the same device as logit_outputs
    # result_tensor = torch.zeros(logit_outputs.shape[0], num_score_fns, device=logit_outputs.device)
    result_tensor = torch.zeros(num_score_fns, logit_outputs.shape[0], device=logit_outputs.device)

    # Loop over each score function
    for fn_idx, score_fn in enumerate(score_fns):
        for idx in range(logit_outputs.shape[0]):
            # hack to handle a function that needs an extra arg
            if score_fn.__name__ == surprisal.__name__:
                res = surprisal(logit_outputs[idx], orig_token_ids[idx])
            else:
                res = score_fn(logit_outputs[idx])
            result_tensor[fn_idx, idx] = res

    return result_tensor


def process_sent_exp6(
        # file_id: str,
        example_id: int,
        sent: str,
        score_fns: List[ScoreFnTensor],
        # topk_logits_to_keep: int = 20,
)-> MLMResultForSentenceExp6:
    """
    Process a given sentence

    Args:
        file_id (str): The file id of the file from which sentence came
        sent_idx_in_file (int): The index of the sentence in file
        sent (str): The sentence to process.
        score_fns: List[ScoreFn]: A list of scoring functions to run on each masked position in the sentence
        topk_logits_to_keep (int): The number of top predictions to keep for fills for each word pos in the sentence

    Takes a sentence (a line in file)
    - create an MLMSentObject to collect results
    - for each line
    - produce a batch
    - run the batch
    - deprocess / postprocess
    - write output

    At present, batches are done at the sentence level and thus will never contain more than num_word_in_sent entries
    """

    # a wrapper around the sentence we're going to process so that we don't recalculate tokens, etc for each word
    entry_for_processing = EntryForProcessing(
        example_id,
        sent,
    )

    ###########
    # get all inputs and then stack
    all_inputs_list = []
    # we are going to mask every token
    tok_idcs = list(range(len(entry_for_processing.tokens)))
    # todo: dimension here
    for idx in tok_idcs:
        inputs = entry_for_processing.get_inputs_with_mask_for_token_at_idx(idx)

        # todo(low) - squeeze and stack could be simplified?
        inputs = inputs.squeeze(dim=0)  # squeeze bc inputs get encoded with an extra unsqueezed dimension
        all_inputs_list.append(inputs)
    # note that torch.stack fails if inputs not of same dimension; this is where we would need to pad if we wanted to to span multiple sentences
    all_inputs = torch.stack(all_inputs_list, dim=0)
    ################

    did_error = False
    try:
        # run forward, get logits
        all_inputs.to(device)
        # logit_outputs = get_logits_for_input_batched(all_inputs)
        logit_outputs = mlm.get_logits_for_input(all_inputs)

        #########
        # base "preprocess" the logits:
        # extract logits at the masked token position for indices
        # this has logits for every token, but we want only the logits at the mask position
        # go from (batch, sent_len, vocab_len) to (batch, vocab_len), where
        # position in batch corresponds to the index of the masked word
        # torch.arange gives us 0..batch-1 so that we can select the first row
        logits_for_masked_only = logit_outputs[torch.arange(logit_outputs.size(0)), tok_idcs]
        #########

        #########
        # scores / results:
        # scores will have shape (num_score_fncs, num_toks_in_sent)
        score_tensor = score_exp6(
            logits_for_masked_only,
            entry_for_processing.input_ids.squeeze(dim=0),
            score_fns)
        scores_list = score_tensor.tolist()
    except Exception as e:
        logging.error(e)
        did_error = True

    if did_error:
        mlm_result_for_sentence = MLMResultForSentenceExp6(
            sentence_id=example_id,
            sentence=sent,
            tokens=None,
            scores=None,
            did_error=True
        )
    else:
        mlm_result_for_sentence = MLMResultForSentenceExp6(
            sentence_id=example_id,
            sentence=sent,
            tokens=entry_for_processing.tokens,
            scores=scores_list,
            did_error=False
        )

    return mlm_result_for_sentence


def magpie_run_for_preprocessed(
        input_file: Path,
        output_file: Path,
        line_limit: Optional[int] = None,
):
    """
    For each line in input_file, process and write to output
    """
    # file_id = input_file.name.split(".")[0]
    # with open(input_file, 'r') as f:
    all_inputs_json: List[MagpieEntryForProcessing] = read_from_jsonl(input_file, MagpieEntryForProcessing)
    line_ct = 0
    for _, me in tqdm(enumerate(all_inputs_json)):
        if line_limit is not None and line_ct >= line_limit:
            break
        try:
            result = process_sent_exp6(
                # todo: need to handle chunking
                example_id=me.id,
                sent=me.sent,
                score_fns = [hhi, min_surprisal, surprisal]
            )
            # write
            dump_to_jsonl(result, output_file)
        except Exception as e:
            logging.error(f"While processing {input_file}, {me.id}\n\t{me.sent}\n "
                          f"Exception {e}")
            traceback.print_exc()

def run_magpie():
    # local derived config
    input_dir = data_dir / "corpus_parsed"
    output_dir = data_dir / "output/magpie"

    # run on sub dir
    run_for_sub_dir(
        input_dir,
        output_dir,
        # we are adapting previous code to work with single dir

        # tokenization has not been cleaned
        ["magpie_unclean"],
        magpie_run_for_preprocessed,
        # file_limit_ct=1
    )

# just for running a test
def run_magpie_local():
    # local derived config
    input_dir = data_dir / "corpus_parsed"
    output_dir = data_dir / "output/magpie"

    # run on sub dir
    run_for_sub_dir(
        input_dir,
        output_dir,
        # we are adapting previous code to work with single dir
        ["magpie_test"],
        magpie_run_for_preprocessed,
        # file_limit_ct=1
    )

if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())

    _c = default_config
    print(_c)

    # work around to make sure logging inits... before setting c
    data_dir = _c.data_dir
    runs_log_dir = data_dir / "runs"
    setup_logging(runs_log_dir, write_print_to_log=False)

    # then we set config after logging is set up
    set_config(_c)

    # run_magpie()
