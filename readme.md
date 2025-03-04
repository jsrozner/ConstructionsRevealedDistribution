# todos
- (todo: link arxiv)
- either zip or push data
- affinity as a module

Repository for Constructions are Revealed in Word Distributions

See src/paper for all experiments

# setup
1. `conda create -n cxs_are_revealed python=3.12 pytorch -c pytorch`
2. `pip install -r requirement.txt`
   - you'll need to also install jupyter to run jupyter notebooks
   - and you'll need
       - `pip install git+https://github.com/jsrozner/rozlib-python.git@v.0.1.0`
3. /src needs to be marked as sources root (pycharm eg.), or code needs to be run from that directory
    - if your code editor does not automatically update python path,
    `__init__.py` files might need to be added
    