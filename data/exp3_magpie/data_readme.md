See also data_config.py

MAGPIE_filtered_split_random.jsonl 
- original magpie from github; see src/data_config.py for link

12_29_magpie_unclean_text_files.zip
- zips of chunked text files
- produced by magpie_process.ipynb
- "unclean" means that there may be extra spaces (i.e. no tokenization was cleaned up)

12_29_magpie_unclean_affinities.zip
- jsonl outputs with calculated affinities

other/minicons_scores.jsonl
- not on github
- likelihood scores for all magpie sentences; ultimately not used in paper
- we tested how much ROC AUC improves if excluding "bad" sentences


