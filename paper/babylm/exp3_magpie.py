import logging
import os
import traceback
from pathlib import Path
from typing import Optional, List

import torch
from tqdm import tqdm

from affinity.tokenization import Sentence, MaskedSent
from babylm.config_babylm import BabyLMModelList
from cxns_in_distrib.exp3_magpie.corpus_magpie import MLMResultForSentenceExp6, MagpieEntryForProcessing
from proj.cxs_are_revealed.paper.data_config import BabyLMExp3Magpie
from lib.common.mlm_singleton import get_singleton_scorer
from lib.exp_common.config import default_config, jag39_config
from lib.scoring_fns import ScoreFnTensor, probability
from lib.scoring_fns import surprisal
from lib.utils.utils_run import run_for_sub_dir
from rozlib.libs.common.config_wrapper import set_config
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl, read_from_jsonl
from rozlib.libs.common.logging.logging import setup_logging
from rozlib.libs.library_ext_utils.utils_torch import get_device

# todo: we should not have needed model list here
# see run_exp for how it was handled there
model_list = BabyLMModelList

device = get_device()

input_dir: Optional[Path] = None
output_dir: Optional[Path] = None
_c = None
data_dir = None
runs_log_dir = None

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
            elif score_fn.__name__ == probability.__name__:
                res = probability(logit_outputs[idx], orig_token_ids[idx])
            else:
                # we use only surprisal and probability
                assert False
                # res = score_fn(logit_outputs[idx])
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

    mlm = get_singleton_scorer()
    s = Sentence(sent, allow_non_alignment_in_tokenization=True)

    # todo(now)
    # jdebene model blows up CUDA if seq is too long; other models (with same max length) don't; not sure why todo
    if s.encoding.input_ids.shape[1] >= mlm.tokenizer.model_max_length:
        logging.warning("sentence too long")
        return MLMResultForSentenceExp6(
            sentence_id=example_id,
            sentence=sent,
            tokens=None,
            scores=None,
            did_error=True
        )
    # todo(now)

    # todo ideally we would only mask the words we actually need, but with batching prob makes little diff
    # todo: note that we are allowing multitoken here which may break things downstream; we need to check when we post-process
    all_masked_sents: List[MaskedSent] = [x for x in s.inputs_for_each_word(allow_multi_token=True)]
    # pp(all_masked_sents)
    # inputs_for_each_word guarantees no multiple masks so we can get index 0

    # todo: this will report single tok indices even when something was multitokenized! (make sure we handle in postproces)
    tok_idcs = [ms.masked_token_indices[0] for ms in all_masked_sents]
    ###########
    # get all inputs and then stack
    all_inputs_list = []
    # we are going to mask every token
    for masked_sent in all_masked_sents:
        # todo: check squeeze
        # todo(low) - squeeze and stack could be simplified?
        all_inputs_list.append(masked_sent.input_ids.squeeze(dim=0))
    # note that torch.stack fails if inputs not of same dimension; this is where we would need to pad if we wanted to to span multiple sentences
    all_inputs = torch.stack(all_inputs_list, dim=0)
    ################
    # pp(all_inputs)

    did_error = False
    try:
        # run forward, get logits

        all_inputs.to(device)
        # logit_outputs = get_logits_for_input_batched(all_inputs)
        logit_outputs = mlm.get_logits_for_input(all_inputs)

        # predicted_ids = torch.topk(logit_outputs, 1, dim=-1).indices
        # pp(predicted_ids)
        #########
        # base "preprocess" the logits:
        # extract logits at the masked token position for indices
        # this has logits for every token, but we want only the logits at the mask position
        # go from (batch, sent_len, vocab_len) to (batch, vocab_len), where
        # position in batch corresponds to the index of the masked word
        # torch.arange gives us 0..batch-1 so that we can select the first row
        # some models have shift
        shifted_tok_idcs = [mlm.maybe_shift_tok_idx_for_decoding(t_idx) for t_idx in tok_idcs]
        logits_for_masked_only = logit_outputs[torch.arange(logit_outputs.size(0)), shifted_tok_idcs]
        #########

        #########
        # scores / results:
        # scores will have shape (num_score_fncs, num_toks_in_sent)
        score_tensor = score_exp6(
            logits_for_masked_only,
            # todo check squeeze
            s.encoding.input_ids.squeeze(dim=0)[tok_idcs],
            score_fns)
        scores_list = score_tensor.tolist()
    except Exception as e:
        print(e)
        # traceback.print_exc() #todo(now)
        logging.error(e)
        did_error = True

    if did_error:
        return MLMResultForSentenceExp6(
            sentence_id=example_id,
            sentence=sent,
            tokens=None,
            scores=None,
            did_error=True
        )

    return MLMResultForSentenceExp6(
        sentence_id=example_id,
        sentence=sent,
        # tokens=entry_for_processing.tokens,
        tokens=s.encoding.tokens(),
        scores=scores_list,
        did_error=False
    )


def magpie_run_for_preprocessed(
        input_file: Path,
        output_file: Path,
        line_limit: Optional[int] = None,
):
    """
    For each line in input_file, process and write to output
    """
    # print("Magpie for preprocessed")
    logging.warning(f"running for {input_file}")
    all_inputs_json: List[MagpieEntryForProcessing] = read_from_jsonl(input_file, MagpieEntryForProcessing)
    line_ct = 0
    for _, me in tqdm(enumerate(all_inputs_json)):
        if line_limit is not None and line_ct >= line_limit:
            break
        line_ct += 1
        try:
            result = process_sent_exp6(
                # todo: need to handle chunking
                example_id=me.id,
                sent=me.sent,
                # score_fns = [hhi, min_surprisal, surprisal]
                score_fns = [surprisal, probability]
            )
            # write
            dump_to_jsonl(result, output_file)
            # print(torch.cuda.memory_allocated(0))
        except Exception as e:
            logging.error(f"While processing {input_file}, {me.id}\n\t{me.sent}\n "
                          f"Exception {e}")
            traceback.print_exc()

    logging.warning(f"for file {input_file} processed {line_ct} lines")

def run_magpie():
    # local derived config
    # input_dir = data_dir / "corpus_parsed"
    # output_dir = data_dir / "output/magpie"

    assert input_dir is not None
    assert output_dir is not None

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
    # print("run magpie local")
    assert input_dir is not None
    assert output_dir is not None

    # run on sub dir
    run_for_sub_dir(
        input_dir,
        output_dir,
        # we are adapting previous code to work with single dir
        # ["magpie_test"],
        ["magpie_unclean"],
        magpie_run_for_preprocessed,
        file_limit_ct=1
    )

# adapted from exp3_run_magpie
def exp3_magpie_jag(**kwargs):
    model_idx = kwargs.pop('model_idx')
    # test_idx = kwargs.pop('test_idx')
    assert model_idx is not None
    model_name = list(model_list.keys())[model_idx]
    model_short_name = model_list[model_name]

    print("Current Working Directory:", os.getcwd())

    global data_dir, runs_log_dir, _c, input_dir, output_dir
    _c = jag39_config
    print(_c)

    # work around to make sure logging inits... before setting c
    data_dir = _c.data_dir
    runs_log_dir = data_dir / "runs"


    # input_dir = BabyLMExp3Magpie._exp3_magpie (MAC)
    input_dir = data_dir
    output_dir = data_dir / f"output/magpie/{model_short_name}"

    setup_logging(runs_log_dir, write_print_to_log=False)

    # then we set config after logging is set up
    set_config(_c)

    run_magpie()

def exp3_magpie_local(**kwargs):
    # print("MAGPIE local")
    model_idx = kwargs.pop('model_idx')
    # test_idx = kwargs.pop('test_idx')
    assert model_idx is not None
    model_name = list(model_list.keys())[model_idx]
    model_short_name = model_list[model_name]

    print("Current Working Directory:", os.getcwd())

    global data_dir, runs_log_dir, _c, input_dir, output_dir
    _c = default_config
    print(_c)

    # work around to make sure logging inits... before setting c
    data_dir = _c.data_dir
    runs_log_dir = data_dir / "runs"

    input_dir = BabyLMExp3Magpie._exp3_magpie
    output_dir = data_dir / f"output/magpie/{model_short_name}"

    # then we set config after logging is set up
    setup_logging(runs_log_dir, write_print_to_log=False)
    set_config(_c)

    run_magpie_local()
