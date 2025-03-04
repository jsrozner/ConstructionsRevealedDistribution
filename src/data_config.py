import os
from pathlib import Path

def current_file_dir(f):
    return os.path.dirname(os.path.abspath(f))  # pyright: ignore [reportUndefinedVariable]

DATADIR = Path(current_file_dir(__file__)) / "../data"


class Exp1Zhou:
    _exp1_zhou = DATADIR / "exp1_zhou"

    ##########
    # Exp1 - zhou CEC vs EAP/AAP
    # original xlsx from leonie of zhou paper
    zhou_original_xlsx = _exp1_zhou / "leonie_so_that_construction.xlsx"
    # all 5 sentence types for each sentence (note we need only the "O", original, type)
    zhou_preprocessed_all_sents = _exp1_zhou / "corpus_leonie_all_processed.txt"
    # outputs for global affinities using surprisal
    zhou_global_affinities_surprisal = _exp1_zhou/"affinities_corpus_leonie_all_surprisal.jsonl"
    # # todo outputs for local affinities and global (uses hhi and euclidean)
    zhou_affinities_hhi_euclid = _exp1_zhou/"corpus_leonie_all_hhi_and_euclid.jsonl"

    ##########
    # Exp1 - multithat
    _exp1_multithat = DATADIR / "exp1_multithat"
    cec_multithat_rozner = _exp1_multithat / "multithat_rozner.csv"
    cec_multithat_input = _exp1_multithat / "multithat_rozner_text_only.txt"
    cec_multithat_affinities = _exp1_multithat / "multithat_rozner_text_only.jsonl"

class Exp2Cogs:
    _exp2_cogs = DATADIR / "exp2_cogs"

    # original cogs data downloaded from CoGs repo
    # https://github.com/H-TayyarMadabushi/Construction_Grammar_Schematicity_Corpus-CoGS/blob/main/Dataset/CoGs.csv
    original_csv = _exp2_cogs / "CoGs.csv"

    # original parsed version (no modifications)
    cogs_parsed = _exp2_cogs / "cogs_parsed.csv"

    # final version after cleaning up any errors etc
    cogs_parsed_final = _exp2_cogs / "cogs_parsed_final.csv"

class Exp3Magpie:
    _exp3_magpie = DATADIR / "exp3_magpie"

    # original magpie, downlaoded from
    # https://github.com/hslh/magpie-corpus
    original_magpie = _exp3_magpie / "MAGPIE_filtered_split_random.jsonl"

    # magpie affinities - directory of jsonl
    magpie_affinity_dir = _exp3_magpie / "12_29_magpie_unclean_affinities"

    # other; can ignore - minicons data; not on github
    magpie_minicons_scores = _exp3_magpie/"other/01_02_magpie_minicons_scores.jsonl"

    #### if rerunning generation
    # directory for chunked text files
    magpie_text_file_dir = _exp3_magpie / "magpie_text"

class Exp4NPN:
    _exp4_npn = DATADIR / "exp4_npn"

    npn_gpt_outputs = _exp4_npn / "npn_gpt4_generations.jsonl"
    npn_outputs_for_human_acceptability_csv = _exp4_npn / "npn_outputs_for_human_acceptability.csv"
    npn_acceptability_ratings_csv = _exp4_npn / "npn_ratings_cory.csv"