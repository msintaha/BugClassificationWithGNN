## Bug Patterns' Graphs
https://drive.google.com/file/d/1h8JQwkDN2E7QrHH-YgVYscMF1T2xa54E/view?usp=sharing 

## Installation instructions

- Create a virtualenv using pyenv of python 3.7.1
```
> pip install virtualenv
> virtualenv venv
> source venv/bin/activate
```
- Run `pip3 install -r requirements.txt` to install packages inside the virtualenv
- Unzip ALL the `tar.gz` files

## How to run

There are 3 args that you need to set. The `--bug_type`, `--use_deepbugs_embeddings` and `--dataset_size` in order to run the classifiers. The arg choices are listed in the respective classifier files. To run with full dataset, replace `--dataset_size=mini` with `--dataset_size=full`

### GAT Classifier on Homogenous graphs for each bug pattern
- `python3 gat_classifier.py --bug_type=incorrect_binary_operand --use_deepbugs_embeddings=True --dataset_size=mini`
- `python3 gat_classifier.py --bug_type=incorrect_binary_operator --use_deepbugs_embeddings=True --dataset_size=mini`
- `python3 gat_classifier.py --bug_type=swapped_args --use_deepbugs_embeddings=True --dataset_size=mini`

### GCN Classifier on Homogenous graphs for each bug pattern
- `python3 gcn_classifier.py --bug_type=incorrect_binary_operand --use_deepbugs_embeddings=True --dataset_size=mini`
- `python3 gcn_classifier.py --bug_type=incorrect_binary_operator --use_deepbugs_embeddings=True --dataset_size=mini`
- `python3 gcn_classifier.py --bug_type=swapped_args --use_deepbugs_embeddings=True --dataset_size=mini`

### RGCN Classifier on Heterogenous graphs
In heterographs, the classifier has to be run separately for each bug pattern.
- For incorrect binary operand related bugs: `python3 rgcn_classifier_heterographs.py --bug_type=incorrect_binary_operand --use_deepbugs_embeddings=True --dataset_size=mini`
- For incorrect binary operator related bugs: `python3 rgcn_classifier_heterographs.py --bug_type=incorrect_binary_operator --use_deepbugs_embeddings=True --dataset_size=mini`
- For swapped args bug: `python3 rgcn_classifier_heterographs.py --bug_type=swapped_args --use_deepbugs_embeddings=True --dataset_size=mini`
