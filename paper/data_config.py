import os
import warnings
from pathlib import Path

def current_file_dir(f):
    return os.path.dirname(os.path.abspath(f))  # pyright: ignore [reportUndefinedVariable]

DATADIR = Path(current_file_dir(__file__)) / "../data"
DATADIRBABY = Path(current_file_dir(__file__)) / "../data_babylm"

# note that babyLM and original paper data config are in the same file (this one);

class Exp1Zhou:
    _exp1_zhou = DATADIR / "exp1_zhou"

    ##########
    # Exp1 - zhou CEC vs EAP/AAP
    # original xlsx from leonie of zhou paper
    zhou_original_xlsx = _exp1_zhou / "leonie_so_that_construction.xlsx"

    # processed zhou data (one sentence per line)
    # this file contains all 5 sentence types for each sentence as described in zhou
    # but our research uses only the 0 type - the original without modifications
    # this presented some minor challenges when we re-ran the EAP/AAP experiments
    # produced in make_all_zhou_data.py
    zhou_preprocessed_all_sents = _exp1_zhou / "corpus_leonie_all_processed.txt"

    # processed zhou data (one sentence per line)
    # unlike above, this contains only the 0 sentence
    #   produced in make_all_zhou_data.py
    zhou_preprocessed_core = _exp1_zhou / "corpus_leonie_processed_core.txt"

    ####
    # zhou data with affinities

    # outputs for global affinities using surprisal (no local affinities)
    zhou_global_affinities_surprisal = _exp1_zhou/"affinities_corpus_leonie_all_surprisal.jsonl"

    # todo(nc)
    #   uses probabilities and JSD scores
    # # todo outputs for local affinities and global (uses hhi and euclidean)
    # todo: should use probability / surprisal and JSD; rerun relabel
    zhou_affinities_hhi_euclid = _exp1_zhou/"corpus_leonie_all_hhi_and_euclid.jsonl"

    zhou_all_core_outputs = _exp1_zhou/"corpus_leonie_probs_and_jsd.jsonl"

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

    # infinigram saved frequencies for v1/v2 dataset (same nouns)
    npn_roberta_infinigram = _exp4_npn / "npn_infinigram_roberta.db"

    ######
    # v1 original sample for primary paper (note some generations failed bc of prompt bug)
    # includes gpt outputs before bad words were removed; not published to public repo
    npn_gpt_outputs_unclean = DATADIR / "../data_other/exp4_npn/npn_gpt4_generations_before_cleaning.jsonl"
    # with bad words removed
    npn_gpt_outputs = _exp4_npn / "npn_gpt4_generations.jsonl"

    npn_outputs_for_human_acceptability_csv = _exp4_npn / "npn_outputs_for_human_acceptability.csv"
    npn_acceptability_ratings_csv = _exp4_npn / "npn_ratings_cory.csv"
    ########

    # fix data v2 - rerun gpt generations for failures
    npn_gpt_outputs_v2_fixed = _exp4_npn / "npn_gpt4_generations_rerun.jsonl"
    npn_outputs_for_human_acceptability_csv_v2 = _exp4_npn / "npn_outputs_for_human_acceptability_v2_resample.csv"
    npn_acceptability_v2 = _exp4_npn / "npn_ratings_cory_v2_final.csv"

    # fix data v3 with only infinigram freq = 0
    npn_gpt_outputs_v3_zero_freq = _exp4_npn / "npn_gpt4_generations_v3_zero_freq.jsonl"
    npn_outputs_for_human_acceptability_csv_v3 = _exp4_npn / "npn_outputs_for_human_acceptability_v3_zero_freq.csv"
    npn_acceptability_v3= _exp4_npn / "npn_ratings_cory_v3_final.csv"

class BabyLMExp3Magpie:
    _exp3_magpie = DATADIR / "exp3_magpie"

    magpie_text_file_dir = _exp3_magpie / "magpie_unclean"

    # new output dir for babyLM outputs
    _exp3_magpie_cluster_out = DATADIRBABY / "magpie"

class BabyLMExp6NPN:
    _npn = DATADIRBABY / "exp6_npn"

    # infinigram data for babylm dataset (this dataset is shared vocab, I think)
    npn_infingram = _npn / "npn_infinigram.db"

    # babyLM generations using joint vocab across all models
    npn_gpt_outputs = _npn / "npn_gpt4_generations_gpt_bert.jsonl"

    npn_outputs_for_human_acceptability_csv = _npn / "npn_outputs_for_human_acceptability.csv"
    npn_acceptability_ratings_csv = _npn / "npn_ratings_cory_babylm.csv"

    # babyLM gpt-bert-big saved counts for our npns
    # todo: where did we set this up
    npn_gpt_bert_big_npn_counts = _npn / "npn_babylm_cts_gptbb.txt"

