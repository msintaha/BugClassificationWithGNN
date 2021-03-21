## Installation instructions

- Create a virtualenv using pyenv of python 3.7.1
- Run `pip3 install -r requirements.txt` to install packages inside the virtualenv
- Unzip ALL the `tar.gz` files

## How to run

- `python3 code_dataset_mini.py --bug_type=incorrect_binary_operator --use_deepbugs_embeddings=True` or `python3 code_dataset_large.py --bug_type=incorrect_binary_operator --use_deepbugs_embeddings=True` will create an instance of the dataset, simply import the dataset class and feed it to the module or paste the classifier code below the dataset creation for now (after line 339)
- Replace `--bug_type` with either `incorrect_binary_operator`, `incorrect_binary_operand` or `swapped_args` to train the model to identify either of the bug types. To use random embeddings, just pass `False` to the arg `--use_deepbugs_embeddings` in the command. 
