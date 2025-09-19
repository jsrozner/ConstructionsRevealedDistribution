# Constructions are Revealed in Word Distribution
Repository for work in [Constructions are Revealed in Word Distributions
(arxiv)](https://arxiv.org/abs/2503.06048)

- See cxs_are_revealed/paper for all experiments
- See example.ipynb for how to play with the affinity methods

# Setup
1. `conda env create -n cxs_are_revealed -f environment.yml
   - you'll need to also install jupyter to run jupyter notebooks
   - and you'll need
       - `pip install git+https://github.com/jsrozner/rozlib-python.git@v.0.1.2`
       - note that the conda install should include all packages required by rozlib
3. environment comments:
   - Because of directory structure, certain folders need to be included in python path
       - in, e.g., pycharm, you can mark /paper and /src as srcs root
       - otherwise you might need to add them to your python path
       - If you don't have pycharm or jupyter doesn't behave:
           - You may want to run your jupyter server from the /src directory since all
           imports are relative to /src 
           - You could alternatively tweak python_path to make sure that src is treated
            as an importable module
    