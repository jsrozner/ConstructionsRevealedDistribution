from __future__ import annotations

from lib.exp_common.config import get_config
from lib.exp_common.mlm_result_for_sentence import MLMResultForSentence
from rozlib.libs.library_ext_utils.utils_torch import get_device
from lib.exp_common.corr_matrix import get_scores
from lib.distr_diff_fcns import euclidean_distance
from lib.scoring_fns import hhi

config = get_config()
device = get_device()

def process_sent_affinity(
        file_id: str,
        sent_idx_in_file: int,
        sent: str,
        score_fn = hhi,
        calculate_affinities=True,
        dist_dff_fn=euclidean_distance
        # mlm: MLMScorer,
        # score_fns: List[ScoreFn],
        # topk_logits_to_keep: int = 20,
)-> MLMResultForSentence:
    """
    Process a given sentence

    Args:
        file_id (str): The file id of the file from which sentence came
        sent_idx_in_file (int): The index of the sentence in file
        sent (str): The sentence to process.
        mlm (MLMScorer): The MLM scorer
        score_fns: List[ScoreFn]: A list of scoring functions to run on each masked position in the sentence
        # topk_logits_to_keep (int): The number of top predictions to keep for fills for each word pos in the sentence

    """
    # todo: we could optimize by using the exp1 gpu approach -> we were lazy and did not

    # todo: support perturbations
    subs_list = ["<mask>"] * len(sent.split())
    (scores,
     new_sents,
     multi_tok_indices,
     _sent_word_list,
     hhis,
     _preds,
     _probs,
     _actual_subs) = (
        get_scores(sent,
                   subs_list,
                   score_fn=score_fn,
                   calculate_affinities=calculate_affinities,
                   dist_diff_fn=dist_dff_fn
                   ))

    # this is where we accumulate results for writing to file
    mlm_result_for_sentence = MLMResultForSentence(
        file_id=file_id,
        sentence_id=sent_idx_in_file,
        sentence=sent,
        # todo: we should do tensor -> list here
        # todo: note that this might not be HHI if another score fn is passed
        hhi_scores=hhis,
        multi_tok_indices=multi_tok_indices,
        perturbed_sentences=new_sents,  #perturbations accumulated here
        score_matrix_distribution=scores.tolist()
        # todo: also add a score matrix using hhi?
    )

    return mlm_result_for_sentence
