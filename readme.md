# Constructions are Revealed in Word Distribution
Repository for work in [Constructions are Revealed in Word Distributions
(arxiv)](https://arxiv.org/abs/2503.06048)

- See cxs_are_revealed/paper for all experiments
- See example.ipynb for how to play with the affinity methods

# Setup
1. `conda create -n cxs_are_revealed python=3.12 pytorch -c pytorch`
2. `pip install -r requirement.txt`
   - you'll need to also install jupyter to run jupyter notebooks
   - and you'll need
       - `pip install git+https://github.com/jsrozner/rozlib-python.git@v.0.1.1`
3. environment comments:
   - Because of directory structure, you might need to add to python exec path
     - cxs_are_revealed/paper and cxs_are_revealed/src
   - Alternatively, in pycharm (or perhaps vscode), 
   you can mark these folders as sources root 
   - If you don't have pycharm or jupyter doesn't behave:
       - You may want to run your jupyter server from the /src directory since all
       imports are relative to /src 
       - You could alternatively tweak python_path to make sure that src is treated
        as an importable module
    