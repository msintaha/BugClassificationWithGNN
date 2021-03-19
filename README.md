## Installation instructions

- Create a virtualenv using pyenv of python 3.7.1
- Run `pip3 install -r requirements.txt` to install packages inside the virtualenv
- Unzip ALL the `tar.gz` files

## How to run

- `python3 code_dataset_mini.py` or `python3 code_dataset_large.py` will create an instance of the dataset, simply import the dataset class and feed it to the module or paste the classifier code below the dataset creation for now (after line 339)
- Replace line 412, 413 with `use_deepbugs_embeddings=True` to use word2vec embeddings. If it is false, we will use random pytorch embeddings
