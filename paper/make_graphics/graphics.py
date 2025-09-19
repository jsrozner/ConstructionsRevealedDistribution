from __future__ import annotations

import os
from pprint import pp
from typing import Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt

from rozlib.libs.plotting.plotting import ibm_colors2
from rozlib.libs.plotting.utils_latex_matplot import FigSaver


class Colors:
    cec_clist = [ibm_colors2[x] for x in [0,2,4]]
    rectangle_color = plt.cm.Dark2(3)  # pyright: ignore [reportAttributeAccessIssue]

def make_fig_saver():
    fig_path = "/Users/jsrozner/docs_local/research/proj_code/rozner-mono-cxs-main/proj/cxs_are_revealed/supplemental/figs"
    return FigSaver(fig_path)


def ensure_tex_in_path() -> None:
    """
    On macOS with MacTeX, latex binaries live in /Library/TeX/texbin.
    Add it to PATH if missing so Matplotlib can find LaTeX.
    """
    texbin = "/Library/TeX/texbin"
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if texbin not in path_parts and os.path.exists(texbin):
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + texbin


# todo move to rozlib
"""
Fix for getting latex formatting in tex
https://stackoverflow.com/questions/69613691/matplotlib-textcolor-doesnt-show-any-colors
https://github.com/matplotlib/matplotlib/issues/6724

Other related
padding: https://stackoverflow.com/questions/42281851/how-to-add-padding-to-a-plot-in-python
bold font: https://tex.stackexchange.com/questions/2783/bold-calligraphic-typeface
"""
def configure_matplotlib_for_latex(fontsize: int = 14) -> None:
    """
    Enable LaTeX text rendering and load xcolor.
    """
    # todo plt / mpl seem the same
    # todo pgf.preamble vs text.latex.preamble
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            # Load xcolor so \textcolor and rgb model are available.
            # Note: 'text.latex.preamble' is supported, though Matplotlib may warn.
            # "text.latex.preamble": r"\usepackage{xcolor}\n\usepackage{amsmath}\n\usepackage{color}",
            "axes.titlesize": fontsize,
        }
    )
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        # 'pgf.preamble': r'\usepackage{color}\usepackage{dashrule}',
        'pgf.preamble': r'\usepackage{xcolor}',
        'text.usetex': True,
        # 'text.latex.preamble':  r'\usepackage{color}\usepackage{dashrule}',
        'text.latex.preamble':  r'\usepackage{xcolor}',
    })
    preamble_parts = [
        r"\usepackage[dvipsnames]{xcolor}",
        r"\usepackage{amsmath}",
        r"\usepackage{bm}",
        r"\definecolor{ibmBlue}{HTML}{648FFF}",
        r"\definecolor{ibmOrange}{HTML}{FFB000}",
    ]
    # note \n or space both work for join
    preamble = "\n".join(preamble_parts)
    # mpl.rcParams["text.usetex"] = True
    mpl.rcParams["pgf.preamble"] = mpl.rcParams["text.latex.preamble"] = preamble
    pp(mpl.rcParams["pgf.preamble"])


def demo_colored_text(
    text: str,
    figsize: Tuple[float, float] = (5.0, 2.0),
    dpi: int = 200,
        fontsize = 14,
    outpath: str | None = "test.png",
) -> None:
    """
    Render a LaTeX-colored text string with Matplotlib.

    Args:
        text: A LaTeX string using \textcolor{<model>}{...} (requires xcolor).
        figsize: Figure size in inches.
        dpi: Output resolution.
        outpath: If provided, save the figure to this path (e.g., 'example.png').
    """
    print(text)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, backend='pgf', dpi=dpi, transparent=True,
                     bbox_inches="tight"
        # pad_inches=1,   # padding already applied via expanded bbox
                    )

        # fig.savefig(outpath, bbox_inches="tight", dpi=dpi)
    plt.show()
