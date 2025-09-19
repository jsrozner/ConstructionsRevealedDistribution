Code for section 4.2 - multithat

## data / preprocessing
See exp1_multithat_preprocess.ipynb
multithat dataset 
- is in data/exp1_multithat/multithat_rozner.csv
- cluster / input for affinities is multithat_rozner_text_only.txt

## get affinities 
- exp1_run_multithat.py will produce affinities (e.g. on a cluster) 
- outputs are in data/exp1_multithat/multithat_rozner_text_only.jsonl

## analysis 
exp1_multithat_analysis.ipynb
- checks that the "cec" 'that' is always the highest affinity with so
- code provided to check any additional sentence that one might want to examine