from collections import defaultdict, Counter
from pathlib import Path
from pprint import pp
from typing import List, Tuple, NamedTuple, Dict

from nltk import word_tokenize

from affinity.corr_matrix_new import get_logits_for_masked_sent, compute_surprisal_for_logits
from affinity.tokenization import Sentence, MultiTokenException
from lib.common.corr_matrix_common import top_k_preds_for_logits
from lib.scoring_fns import probability

from proj.cxs_are_revealed.paper.proj_common.npn_dataset_generation.npn_utils import HumanAnnotation, GPTOutput, GPTOutput_with_nounrep, filter_outputs, count_for_stats
from rozlib.libs.common.data.utils_dataclass import read_csv_to_dataclass
from rozlib.libs.common.data.utils_jsonl import read_from_jsonl


def read_judgements(npn_judgement_file: Path) -> List[HumanAnnotation]:
    all_judgements = read_csv_to_dataclass(HumanAnnotation, npn_judgement_file)
    for j in all_judgements:
        j.id = int(j.id)
        j.rating = int(j.rating)
        j.sentence = j.sentence.strip()
    return all_judgements


def filter_with_judgements(judgements_aligned: List[HumanAnnotation],
                           gpt_outputs_with_id: List[Tuple[int, GPTOutput]],
                           min_score
                           ) -> List[GPTOutput]:

    ret: List[GPTOutput] = []
    # go[0] is the index and it maps to a given aligned judgement
    for go in gpt_outputs_with_id:
        idx = go[0]
        j_aligned = judgements_aligned[idx]
        assert j_aligned.sentence == go[1].output.strip()
        if j_aligned.rating < min_score:
            continue
        ret.append(go[1])
    return ret


def get_npn_data(
        npn_gpt_output_file: Path,
        npn_judgement_file: Path | None,
        min_acceptability: int | None,
        output_has_noun_rep = False,   # todo: typing, should be a dataclass instance
        do_filter_gpt_generations = True
) -> List[GPTOutput]:
    """
    todo: Supports output_has_noun_rep (set True for original paper) for our rebuttal analysis
    # todo - we should update the lib files for original paper

    Args:
        npn_gpt_output_file:
        npn_judgement_file:
        output_has_noun_rep:
        min_acceptability:

    Returns:

    """

    ###########
    # read in gpt generations
    # reproduce previous filtering / cleaning
    if output_has_noun_rep:
        # fix the typing
        gpt_out_noun_rep: List[GPTOutput_with_nounrep] = read_from_jsonl(npn_gpt_output_file, GPTOutput_with_nounrep)
        # hacks
        for g in gpt_out_noun_rep:
            g.noun = g.noun.str_rep.strip()
        gpt_outputs: List[GPTOutput] = gpt_out_noun_rep
    else:
        gpt_outputs: List[GPTOutput] = read_from_jsonl(npn_gpt_output_file, GPTOutput)
    print("reading from gpt_output_file - before filtering", len(gpt_outputs))

    ###########
    # filter to make sure target string is present
    if do_filter_gpt_generations:
        gpt_outputs_after = filter_outputs(gpt_outputs, print_errors=True)
    else:
        gpt_outputs_after = gpt_outputs
    print(f"after filtering (did filter = {do_filter_gpt_generations}), {len(gpt_outputs_after)}")
    print(len(gpt_outputs_after))

    gpt_outputs_with_id = [(idx, go) for idx, go in enumerate(gpt_outputs_after)]
    for go in gpt_outputs_with_id:
        go[1].output = go[1].output.strip()

    ###########
    # maybe filter to human judgements
    if npn_judgement_file is None:
        assert min_acceptability is None, "npn judgement file not given but min acceptability given"
        return gpt_outputs_after
    assert min_acceptability is not None, "npn judgement file given but no min acceptability given"

    # read in human judgements
    all_judgements = read_judgements(npn_judgement_file)
    judgements_aligned = sorted(all_judgements, key=lambda x: x.id)
    count_for_stats(judgements_aligned, gpt_outputs_with_id)

    if len(judgements_aligned) != len(gpt_outputs_with_id):
        print(f"length mismatch: {len(judgements_aligned)} != {len(gpt_outputs_with_id)}")
        print(set([x.id for x in all_judgements]).difference(set([x[0] for x in gpt_outputs_with_id])))

    assert len(judgements_aligned) == len(gpt_outputs_with_id),\
        f"length mismatch: {len(judgements_aligned)} != {len(gpt_outputs_with_id)}"

    for judgement, gpt_out in zip(judgements_aligned, gpt_outputs_with_id):
        # gpt_output_with_id[0], the index, matches judgement.id
        if judgement.sentence != gpt_out[1].output.strip():
            print('mismatch')
            pp(judgement)
            pp(gpt_out[1])
            raise Exception("judgement and gpt output do not match")

    gpt_outputs_acceptable = filter_with_judgements(judgements_aligned,
                                                    gpt_outputs_with_id,
                                                    min_acceptability)
    return gpt_outputs_acceptable


class NPNResult(NamedTuple):
    orig_gpt: GPTOutput
    tokenized_sent: str
    prep: str
    score: float
    fills: List[str]


def compute_scores(
        npns: List[GPTOutput],
        allow_case_mismatch=True
) -> Tuple[Dict[str, List[float]], int, int, List[NPNResult]]:
    scores: Dict[str, List[float]] = defaultdict(list)
    ct_err = 0
    ct_multi = 0
    results: List[NPNResult] = []
    for r in npns:
        sent_tokenized_list = word_tokenize(r.output)
        sent_tokenized_with_spaces = " ".join(sent_tokenized_list)   # this will be tokenized version (spaces bw punctuation)

        # compute the target offsets and words
        tgt_str = f"{r.noun} {r.prep} {r.noun}"
        if allow_case_mismatch:
            sent_tokenized_with_spaces = sent_tokenized_with_spaces.lower()
            tgt_str = tgt_str.lower()
        phrase_idx_start = sent_tokenized_with_spaces.find(tgt_str)
        assert phrase_idx_start >= 0, f"tgt str not found ({tgt_str}) not in \n\t({sent_tokenized_with_spaces})"
        phrase_idx_end = phrase_idx_start + len(tgt_str)
        phrase_parts = sent_tokenized_with_spaces[phrase_idx_start: phrase_idx_end].split(" ")
        for pp, expected in zip(phrase_parts, [r.noun, r.prep, r.noun]):
            assert pp == expected, f"[{r.output}]: {pp} != {expected}"

        # compute target offsets and target words
        tgt_word_offsets = []
        start_idx = phrase_idx_start
        for w in phrase_parts:
            w_len = len(w)
            assert sent_tokenized_with_spaces[start_idx: start_idx + w_len] == w, f"{tgt_str[start_idx: start_idx + w_len]} != {w}"
            if w not in ['upon', 'after', 'by', 'to']:
                tgt_word_offsets.append((start_idx, start_idx + w_len))
            start_idx += w_len + 1  # add 1 for space
        target_words = [sent_tokenized_with_spaces[x[0]: x[1]] for x in tgt_word_offsets]

        for w, expected in zip(target_words, [r.noun, r.noun]):
            assert w == expected, f"{w} != {expected}"

        # run for the sentence -> will process two nouns
        try:
            # todo: this is already catching
            s = Sentence(sent_tokenized_with_spaces, allow_non_alignment_in_tokenization=True)
            if s.words_clean is None:
                ct_err += 2
                continue
        except:
            ct_err += 2    # will mean that we lose two nouns
            continue

        for offset, word in zip(tgt_word_offsets, target_words):
            occ_ct = sent_tokenized_with_spaces[:offset[1] + 1].count(word + " ")
            assert occ_ct >= 1, f""

            # will error if multitoken or if multiple occurrences of the word
            try:
                # fix 1 -> 0 index in occ
                masked_sent = s.get_input_with_word_masked(word, occ=occ_ct -1)
            except MultiTokenException as e:
                print(f"multioken: {word}\n\t{r.output}")
                ct_multi += 1
                continue

            logits = get_logits_for_masked_sent(masked_sent)
            fills = top_k_preds_for_logits(logits, 5)
            prob = compute_surprisal_for_logits(
                masked_sent,
                logits,
                probability
            )
            scores[r.prep].append(prob)
            results.append(NPNResult(
                r,
                sent_tokenized_with_spaces,
                r.prep,
                prob,
                fills
            ))

    print(ct_err, ct_multi)
    return scores, ct_err, ct_multi, results


def count_by_prep(gpt_output_list: List[GPTOutput]):
    c = Counter()
    for o in gpt_output_list:
        c[o.prep] += 1
    pp(c)
