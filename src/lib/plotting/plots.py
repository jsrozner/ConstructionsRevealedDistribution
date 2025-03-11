"""
Convenience functions to generate affinity plots for presentation / overleaf
"""
import warnings
from typing import List, Optional, Tuple
from pprint import pp

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from lib.distr_diff_fcns import euclidean_distance, jensen_shannon_divergence
from lib.scoring_fns import hhi_rounded, probability
from lib.exp_common.corr_matrix import get_scores
from lib.plotting.plot_corr_matrix import plot_heatmap
from rozlib.libs.plotting.plotting import add_X_to_plot
from lib.utils.utils_misc import save_fig


def plot_all_affinities(
        sent: str,
        subs_list: Optional[List[str]] = None,
        use_euclid: bool = False,
        use_probability: bool = True,
        num_preds = 5,
        do_make_local_aff_heatmap= True,
        cmap="Grays",
        output_file_for_global_aff=None,
        max_color_percent_for_global_aff: Optional[float]=None,
        do_print = False
) -> List[float]:
    """
    Plots local and global affinities

    Args:
        sent: the sentence to be plotted
        subs_list: Subs to use when calculating local affinities. If not given, will default to <mask> for all
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

    if subs_list is None:
        subs_list = ["<mask>"] * len(sent.split())

    local_aff_scores, new_sents, multi_tok_indices, sent_word_list, global_aff_scores, preds, probs, actual_subs = get_scores(
        sent,
        subs_list,
        score_fn = score_fn ,
        subs_method="mask",
        num_preds=num_preds,
        dist_diff_fn=dist_fn,
        calculate_affinities=do_make_local_aff_heatmap
    )

    if do_print:
        print("*" * 20)
        print(global_aff_scores)
        print(f"multitoken indices are {multi_tok_indices}")
        pp(preds)
        pp(probs)

    if do_make_local_aff_heatmap:
        plot_heatmap(local_aff_scores,
                     sent_word_list,
                     # actual_subs,
                     None,
                     cmap=cmap,
                     title="Distributional Change under Perturbation",
                     )

    generate_word_score_image(
        sent_word_list,
        global_aff_scores,
        invert_scores=False,
        assert_all_scores_0_to_1=not use_probability,
        output_file=output_file_for_global_aff,
        max_color_percent=max_color_percent_for_global_aff
    )

    return global_aff_scores


def plot_one_global_aff(
        input: List[float],
        xlabels: List[str],
        figsize_if_plotting_single=(3, 3),
        ax = None
) -> Optional[Figure]:
    """
    Plots a single global affinity. (Single dimension heatmap with words)
    - if called with ax=None, then will create a single plot
    - pass with ax to add to multiplot (e.g., Called for each plot by :func:`plot_multiple_global_aff`)

    Args:
        input:
        xlabels:
        figsize_if_plotting_single:
        ax: Pass if adding to multiplot; otherwise leave as None to have a plot created
    Returns:

    """
    if ax is None:
        print("making subplots")
        fig, ax = plt.subplots(figsize=figsize_if_plotting_single)
    else:
        fig = None
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)  # Hides tick marks
    # input is single dimensions; imshow needs 2d list
    ax.imshow([input], cmap="Grays", vmin=0, vmax=1)

    return fig


def plot_multiple_global_aff(
        data: List[Tuple[List[float], List[str]]],
        overall_fig_size = (3,3),
        x_locs: Optional[List[List[int]]] = None,
        add_scores_on_top=False,
        add_scores_on_top_font_size=12
) -> Figure:
    """
    Plot multiple global affinities:
    - each heatmap is a single dimension heatmap

    Args:
        data: List of Tuples of scores (List[float]) and words (List[str])
        overall_fig_size:
        x_locs: where to insert X's (for aligning plots with some words missing)

    Returns:

    """
    num_rows = len(data)
    fig, axes = plt.subplots(num_rows, 1, figsize=overall_fig_size, squeeze=False)
    axes = axes.ravel()

    data_lens = [len(d[1]) for d in data]
    # print(data_lens)
    max_len = max(data_lens)
    per_cube_len = overall_fig_size[0]/max_len
    per_fig_height = overall_fig_size[1]/len(data)

    for i, (d, ax) in enumerate(zip(data, axes)):
        ax.grid(False) # todo: why needed?
        plot_one_global_aff(
            d[0],
            d[1],
            ax=ax,
            figsize_if_plotting_single=(per_cube_len * len(data[0]),
                                        per_fig_height)
        )
        if x_locs:
            for x in x_locs[i]:
                add_X_to_plot(ax, len(d[1]), 1,x,0)
        if add_scores_on_top:
            # print("adding scores")
            # print(d[0])
            for idx, s in enumerate(d[0]):
                ax.text(idx, -0.7, round(s, 2),ha="center", va="center", fontsize=add_scores_on_top_font_size)

    # plt.show()
    return fig

def full_plot_single_sentence(
        s: str,
        add_scores_on_top = True,
        add_scores_on_top_font_size = 5,
        do_print=False,
        plot_local=False,
        use_euclid=False,
        num_preds=0,
):
    """
    Simple convenience function to plot a single dim heatmap for sentence s
    (will strip ending period)

    Args:
        s: sent to plot

    Returns:

    """
    gaf = plot_all_affinities(
        s,
        do_make_local_aff_heatmap=plot_local,
        do_print=do_print,
        use_euclid=use_euclid,
        num_preds=num_preds,
    )
    all_words = s.strip(".").split(" ")
    return plot_multiple_global_aff(
        [(gaf, all_words)],
        x_locs=None,
        add_scores_on_top=add_scores_on_top,
        add_scores_on_top_font_size=add_scores_on_top_font_size,
    )


def generate_word_score_image(
        words: List[str],
        scores: List[float],
        assert_all_scores_0_to_1: bool = True,
        invert_scores = False,
        output_file: Optional[str] = None,
        max_color_percent = None
) -> None:
    """
    Generate a colored image based on word scores.

    Args:
        words (List[str]): List of words to display.
        scores (List[float]): List of scores (between 0 and 1) corresponding to the words.
        output_file (str): File name to save the image. Default is 'word_scores.png'.
    """
    # print(words)
    # print(scores)
    if len(words) != len(scores):
        print(words, len(scores))
        raise ValueError("The length of words and scores must be the same.")

    score_labels = scores.copy()

    # Assert that all scores are in the range [0, 1]
    if not all(0 <= score <= 1 for score in scores):
        if assert_all_scores_0_to_1:
            raise ValueError("All scores must be in the range [0, 1].")
        else:
            warnings.warn(f"Some values are not in [0,1]; min is {min(scores)}; max is {max(scores)}, will truncate to [0,1]")
            scores = list(map(lambda x: max(min(x, 1), 0), scores))


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(len(words), 1))

    # Create a red-scale color gradient (from white to dark red)
    # gradient = plt.cm.Blues(np.linspace(0, 1, 256))
    gradient = plt.cm.Grays(np.linspace(0, 1, 256))
    # reduce intensity
    if max_color_percent:
        assert 0 < max_color_percent < 1
        max_color = 230 * max_color_percent
    else:
        max_color =230
    if invert_scores:
        colors = [gradient[int((1-score) ** 2 * max_color)] for score in scores]
    else:
        colors = [gradient[int(score ** 2 * max_color)] for score in scores]

    # Add words as black text with color bars below, bold and underlined if score > 0.9
    for idx, (word, color) in enumerate(zip(words, colors)):
        if len(word) > 0:
            # allow empty words for a particular image creation for overleaf
            txt = f"{word}\n{round(score_labels[idx], 2)}"
        else:
            txt = ""
        ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=color))  # Color bar
        fontweight = 'bold'
        # textdecoration = 'underline' if scores[idx] > 0.9 else 'normal'
        ax.text(
            # fontsize=10,
            # idx + 0.5, 0.5, txt, va='center', fontsize=13,
            idx + 0.5, -0.5, txt, va='center',
            # rotation="vertical",
            rotation=45,
            ha='right',
            # ha='center', fontsize=12,
            color='black',
            fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.0", facecolor="none", edgecolor="none")
        )
    # Adjust axis limits and turn off axis
    ax.set_xlim(0, len(words))
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save the image
    plt.show()
    if output_file is not None:
        save_fig(fig, output_file)


def generate_word_score_image_new(
        words: List[str],
        scores: List[float],
        assert_all_scores_0_to_1: bool = True,
        invert_scores = False,
        output_file: str = "word_scores.png") -> None:
    """
    Generate a colored image based on word scores.

    Args:
        words (List[str]): List of words to display.
        scores (List[float]): List of scores (between 0 and 1) corresponding to the words.
        output_file (str): File name to save the image. Default is 'word_scores.png'.
    """
    # print(words)
    # print(scores)
    # print(all)
    if len(words) != len(scores):
        raise ValueError("The length of words and scores must be the same.")

    score_labels = scores.copy()

    # Assert that all scores are in the range [0, 1]
    if not all(0 <= score <= 1 for score in scores):
        if assert_all_scores_0_to_1:
            raise ValueError("All scores must be in the range [0, 1].")
        else:
            warnings.warn(f"Some values are not in [0,1]; min is {min(scores)}; max is {max(scores)}, will truncate to [0,1]")
            scores = list(map(lambda x: max(min(x, 1), 0), scores))


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(len(words), 1))

    # Create a red-scale color gradient (from white to dark red)
    # gradient = plt.cm.Blues(np.linspace(0, 1, 256))
    gradient = plt.cm.Grays(np.linspace(0, 1, 256))
    # reduce intensity
    if invert_scores:
        colors = [gradient[int((1-score) ** 2 * 230)] for score in scores]
    else:
        colors = [gradient[int(score ** 2 * 230)] for score in scores]

    # Add words as black text with color bars below, bold and underlined if score > 0.9
    for idx, (word, color) in enumerate(zip(words, colors)):
        if len(word) > 0:
            # allow empty words for a particular image creation for overleaf
            txt = f"{word}\n{round(score_labels[idx], 2)}"
        else:
            txt = ""
        ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=color))  # Color bar
        fontweight = 'bold'
        # textdecoration = 'underline' if scores[idx] > 0.9 else 'normal'
        ax.text(
            idx + 0.5, 0.5, txt, va='center', ha='center', fontsize=10,
            color='black',
            fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.0", facecolor="none", edgecolor="none")
        )
    # Adjust axis limits and turn off axis
    ax.set_xlim(0, len(words))
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save the image
    plt.show()
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.close()
