import statistics
from pathlib import Path
from typing import List, Dict

from proj.cxs_are_revealed.paper.data_config import BabyLMExp6NPN
from proj.cxs_are_revealed.paper.proj_common.npns.npns import get_npn_data, compute_scores, count_by_prep
from proj.cxs_are_revealed.paper.proj_common.npn_dataset_generation.npn_utils import GPTOutput


def filter_npns_from_counts_data(
        input_npns: List[GPTOutput],
        counts_file: Path
) -> List[GPTOutput]:
    # process counts from file
    # see babylm_datasets for where this was generated
    npn_to_count: Dict[str, int] = {}
    with open(counts_file) as f:
        lines = f.readlines()
        for l in lines:
            npn, ct = tuple(l.split(","))
            ct = int(ct)
            npn_to_count[npn] = ct

    # filter the npns
    output_npns = []
    for x in input_npns:
        npn = x.noun + " " + x.prep + " " + x.noun
        if npn_to_count[npn] == 0:
            output_npns.append(x)

    print(f"after filtering to freq = 0: {len(input_npns)} => {len(output_npns)}")
    return output_npns


def exp6_npn(**kwargs):
    # get the data

    all_npns: List[GPTOutput] = get_npn_data(
        npn_gpt_output_file=BabyLMExp6NPN.npn_gpt_outputs,
        output_has_noun_rep=False,
        do_filter_gpt_generations=True,

        # npn_judgement_file=None,
        # min_acceptability=None,
        npn_judgement_file=BabyLMExp6NPN.npn_acceptability_ratings_csv,
        min_acceptability=4,
    )
    print("all npns after acceptability filter")
    count_by_prep(all_npns)

    all_npns = filter_npns_from_counts_data(
        all_npns,
        BabyLMExp6NPN.npn_gpt_bert_big_npn_counts
    )
    print("all npns after freq filter")
    count_by_prep(all_npns)

    all_npns_target_preps = [
        # x for x in all_npns if x.prep in ['upon', 'after']
        x for x in all_npns
    ]
    scores, ct_err, ct_multi, results = compute_scores(
        all_npns_target_preps,
        # todo: check this choice
        allow_case_mismatch=True
    )
    scores_to_return = []
    keys = sorted(scores.keys())
    for k in keys:
        score = statistics.mean(scores[k])
        scores_to_return.append(score)
        print(k, score)

    return (
        *tuple(scores_to_return), ct_err, ct_multi
    )
