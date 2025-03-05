Code for section 5.1 - CoGs

## data
data/exp3_magpie/MAGPIE_filtered_split_random.jsonl
downloaded from github; see data_config.py

## preprocess
magpie_preprocess.ipynb
produces the files in data/exp3_magpie/12_29_magpie_unclean.zip

## produce affinities
- will need to unzip the above
- `python exp3_run_magpie.py` - we ran on cluster

## compute affinities and analysis
See exp3_magpie_analysis.ipynb for roc plot generation

