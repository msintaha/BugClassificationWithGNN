## Installation instructions

- Create a virtualenv using pyenv of python 3.7.1
- Run `pip3 install -r requirements.txt` to install packages inside the virtualenv
- Unzip ALL the `tar.gz` files

## How to run

There are 3 args that you need to set. The `--bug_type`, `--use_deepbugs_embeddings` and `--dataset_size` in order to run the classifiers. The arg choices are listed in the respective classifier files

### GAT Classifier on Homogenous graphs
`python3 gat_classifier.py --bug_type=binOps --use_deepbugs_embeddings=True --dataset_size=mini`

### GCN Classifier on Homogenous graphs
`python3 gcn_classifier.py --bug_type=binOps --use_deepbugs_embeddings=True --dataset_size=mini`

### RGCN Classifier on Heterogenous graphs
- For binary operator related bugs: `python3 rgcn_classifier_heterographs.py --bug_type=binOps --use_deepbugs_embeddings=True --dataset_size=mini`
- For swapped args bug: `python3 rgcn_classifier_heterographs.py --bug_type=swapped_args --use_deepbugs_embeddings=True --dataset_size=mini`
