from lib.exp_common.mlm_result_for_sentence import MLMResultForSentence
from lib.plotting.graph_affinity import make_affinity_graph
from lib.exp_common.corr_matrix import get_info_for_mlm_sentence
import torch
from lib.plotting.plot_corr_matrix import plot_heatmap
# from scoring.exp2_affinity_matrix.dataclass_exp2 import MLMResultForSentence

def plot_from_mlm_result(
        mlm_result: MLMResultForSentence,
        do_plot_heatmap = True,
        omit_adjacent = False
):
    # todo: note that this assumes that we used <mask> for all the subs
    subs_list = ["<mask>"] * len(mlm_result.sentence.split())
    sent_word_list, new_sents_under_sub = get_info_for_mlm_sentence(mlm_result, subs_list)
    for ns1, ns2 in zip(mlm_result.perturbed_sentences, new_sents_under_sub):
        assert ns1 == ns2
    score_tensor = torch.Tensor(mlm_result.score_matrix_distribution)
    if do_plot_heatmap:
        plot_heatmap(
            score_tensor,
            sent_word_list,
            subs_list,
            cmap="Blues"
        )
    make_affinity_graph(
        score_tensor,
        mlm_result.multi_tok_indices,
        sent_word_list,
        mlm_result.hhi_scores,
        omit_adjacent=omit_adjacent
    )
