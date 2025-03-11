from pprint import pp
from typing import Optional, List

from affinity.corr_matrix_new import get_scores_new
from affinity.tokenization import Sentence
from lib.distr_diff_fcns import euclidean_distance, jensen_shannon_divergence
from lib.plotting.plot_corr_matrix import plot_heatmap
from lib.plotting.plots import plot_multiple_global_aff, generate_word_score_image
from lib.scoring_fns import probability, hhi_rounded


def full_plot_single_sentence_new(
        orig_sent: str,
        add_scores_on_top = True,
        add_scores_on_top_font_size = 5,
        do_print=False,
        plot_local=False,
        use_euclid=False,
        num_preds=0,
        normalize=False,
):
    """
    Simple convenience function to plot a single dim heatmap for sentence s
    (will strip ending period)

    Args:
        s: sent to plot

    Returns:

    """
    s = Sentence(orig_sent)
    gaf = plot_all_affinities_new(s,
                                  do_make_local_aff_heatmap=plot_local,
                                  do_print=do_print,
                                  use_euclid=use_euclid,
                                  num_preds=num_preds,
                                  normalize = normalize
                                  )
    return plot_multiple_global_aff(
        [(gaf, s.words_clean)],
        x_locs=None,
        add_scores_on_top=add_scores_on_top,
        add_scores_on_top_font_size=add_scores_on_top_font_size,
    )


def plot_all_affinities_new(
        sent: Sentence,
        use_euclid: bool = False,
        use_probability: bool = True,
        num_preds = 5,
        do_make_local_aff_heatmap= True,
        cmap="Grays",
        output_file_for_global_aff=None,
        max_color_percent_for_global_aff: Optional[float]=None,
        do_print = False,
        normalize = False
) -> List[float]:
    """
    Plots local and global affinities

    Args:
        sent: the sentence to be plotted
        use_euclid: For local affinity: use euclidean distance; else use JSD
        use_probability: For global affinity: use probability; else use HHI
        num_preds: Number of predictions for each fill to print
        do_make_local_aff_heatmap: Whether to calculate local affinities and produce heatmap
        cmap: color map for matplotlib
        output_file_for_global_aff: Output file for global aff; None will not save
        max_color_percent_for_global_aff: scale colors in output
        do_print: whether to print details (multitoken, predictions, probs)

    Returns:
        object: global affinity scores

    """
    if use_euclid:
        dist_fn = euclidean_distance
    else:
        dist_fn = jensen_shannon_divergence

    if use_probability:
        # score_fn = surprisal
        score_fn = probability
    else:
        score_fn = hhi_rounded

    local_affinities, _, multi_tok_indices, global_affinities, preds = (
        get_scores_new(
            sent,
            score_fn = score_fn ,
            num_preds=num_preds,
            calculate_affinities=do_make_local_aff_heatmap,
            dist_diff_fn=dist_fn,
            normalize=normalize
    ))

    if do_print:
        # print("*" * 20)
        print("global affinities:")
        pp([round(x, 3) for x in global_affinities])
        print(f"multitoken indices are:\n {multi_tok_indices}")
        print("Top predictions for each word are:")
        pp(preds)

    if do_make_local_aff_heatmap:
        plot_heatmap(local_affinities,
                     sent.words_clean,
                     # actual_subs,
                     None,
                     cmap=cmap,
                     title="Local Affinities",
                     )

    # generate_word_score_image(
    #     sent.words_clean,
    #     global_affinities,
    #     invert_scores=False,
    #     assert_all_scores_0_to_1=not use_probability,
    #     output_file=output_file_for_global_aff,
    #     max_color_percent=max_color_percent_for_global_aff
    # )

    return global_affinities
