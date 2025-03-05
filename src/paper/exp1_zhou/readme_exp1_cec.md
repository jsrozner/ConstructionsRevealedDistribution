Code for section 4.1 - the CEC vs EAP/AAP classification

## data
data/exp1_zhou/leonie_so_that_construction.xlsx was provided directly by Leonie Weissweiler, 
coauthor on the original Zhou paper and on this paper.

## preprocessing
`python make_all_zhou_data.py`
will produce a text file with all sentences in the Zhou dataset 
- Our code cleans the examples 
- Our code produces each of the 5 sentence types in the Zhou paper, but only
      "O", the original form, is necessary for this experiment 

## compute affinities 
`python exp1_run_mlm.py` will produce affinities (on cluster or local)
- note that only surprisal scores (global affinities) are needed for this section
- our cluster output for global affinities is in affinites_corpus_leonie_all_surprisal.jsonl

## analysis 
exp1_analysis.ipynb

## comments
./other can be ignored
