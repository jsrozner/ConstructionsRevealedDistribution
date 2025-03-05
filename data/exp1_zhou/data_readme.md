leonie_so_that_constructions.xlsx:
    original data (zhou dataset)

corpus_leonie_all_processed.txt:
    all 5 forms of the sentences in the original dataset
    note that only the O (original) form is actually needed for our experiment

affinities_corpus_leonie_all_surprisal:
    - results of running *global* affinity scoring (local affinites were not calculated)
    - results here are with surprisal / probabilities

corpus_leonie_all_hhi_and_euclid:
    - local affinity and global affinity for all sentences
    - this used a different scoring mechanism: HHI and euclidean rather than jensen-shannon-divergence
