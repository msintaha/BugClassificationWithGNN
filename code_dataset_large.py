import json

binOps_training_data_paths = './binOps_training/binOps_training_150.json'
binOps_validation_data_paths = './binOps_eval/binOps_eval_150.json'
calls_training_data_paths = './calls_training/calls_training_150.json'
calls_validation_data_paths = './calls_eval/calls_eval_150.json'


token_vectors = None
type_vectors = None
node_type_vectors = None

binOps_training = None
binOps_eval = None
calls_training = None

with open('./token_to_vector/token_to_vector_150.json') as f:
  token_vectors = json.load(f)

with open('./type_to_vector.json') as f:
  type_vectors = json.load(f)

with open('./node_type_to_vector.json') as f:
  node_type_vectors = json.load(f)

with open(binOps_training_data_paths) as f:
  binOps_training = json.load(f)

with open(calls_training_data_paths) as f:
  calls_training = json.load(f)

with open(binOps_validation_data_paths) as f:
  binOps_eval = json.load(f)

with open(calls_validation_data_paths) as f:
  calls_eval = json.load(f)

### Create graph tuples of positive and negative examples from word2vec embeddings

import dgl
import os
import torch as th
import random

from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from collections import namedtuple


binOps_graph = ([0, 1, 1, 1, 3, 4, 2, 5], [1, 2, 3, 5, 4, 2, 6, 6])
calls_graph = ([0, 1, 1, 2, 2, 3, 5, 6, 4], [1, 2, 5, 5, 3, 4, 6, 7, 7])
operator_embedding_size = 30
name_embedding_size = 200
type_embedding_size = 5
Operand = namedtuple('Operand', ['op', 'type'])
LABELS = {
    'correct_binary_op': 0,
    'incorrect_binary_operator': 1,
    'incorrect_binary_operand': 2,
    'correct_args': 4,
    'swapped_args': 5,
}


class CorrectAndBuggyDataset(DGLDataset):
    def __init__(self, use_deepbugs_embeddings=True, is_training=True, bug_type='incorrect_binary_operator'):
        self.file_to_operands = dict()
        self.all_operators = None
        self.graphs = []
        self.labels = []
        self.use_deepbugs_embeddings = use_deepbugs_embeddings
        self.is_training = is_training
        self.bug_type = bug_type

        super().__init__(name='synthetic')
        
    ## This is for determining all possible operator types to specify the length of operator vector
    def pre_scan_binOps(self, first_data_paths, second_data_paths=[]):
        all_operators_set = set()
        for bin_op in first_data_paths:
            file = bin_op['src'].split(' : ')[0]
            operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op['left'], bin_op['leftType'])
            right_operand = Operand(bin_op['right'], bin_op['rightType'])
            operands.add(left_operand)
            operands.add(right_operand)

            all_operators_set.add(bin_op['op'])
        if second_data_paths == []:
            self.all_operators = list(all_operators_set)
            return

        for bin_op in second_data_paths:
            file = bin_op['src'].split(' : ')[0]
            operands = self.file_to_operands.setdefault(file, set())
            left_operand = Operand(bin_op['left'], bin_op['leftType'])
            right_operand = Operand(bin_op['right'], bin_op['rightType'])
            operands.add(left_operand)
            operands.add(right_operand)

            all_operators_set.add(bin_op['op'])
        self.all_operators = list(all_operators_set)

    
    def generate_random_embedding(self, num_nodes):
        return th.randn(num_nodes, name_embedding_size)
    
    
    def get_tensor_feature(self, data):
        max_len = max([x.squeeze().numel() for x in data])
        # pad all tensors to have same length
        data = [th.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
        # stack them
        return th.stack(data)
        

    def generate_graphs_from_binOps_ast(self):
        num_nodes = 7
        
        dataset = binOps_training if self.is_training else binOps_eval
        for data in dataset:
            left = data['left']
            right = data['right']
            operator = data['op']
            left_type = data['leftType']
            right_type = data['rightType']
            parent = data['parent']
            grand_parent = data['grandParent']
            src = data['src']
            
            if not (left in token_vectors):
                continue
            if not (right in token_vectors):
                continue

            operator_vector = [0] * operator_embedding_size
            operator_vector[self.all_operators.index(operator)] = 1
            correct_vector = [
                th.tensor(node_type_vectors[grand_parent]),
                th.tensor(node_type_vectors[parent]),
                th.tensor(operator_vector),
                th.tensor(type_vectors[left_type]),
                th.tensor(token_vectors[left]),
                th.tensor(type_vectors[right_type]),
                th.tensor(token_vectors[right]),
            ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)
            
            g = dgl.graph(binOps_graph, num_nodes=num_nodes)
            g.ndata['features'] = self.get_tensor_feature(correct_vector)
            self.graphs.append(g)
            self.labels.append(0)
            
            ## Incorrect binary operator
            if self.bug_type == 'incorrect_binary_operator':
                other_operator = None
                other_operator_vector = None

                while other_operator_vector == None:
                    other_operator = random.choice(self.all_operators)
                    if other_operator != operator:
                        other_operator_vector = [0] * operator_embedding_size
                        other_operator_vector[self.all_operators.index(
                            other_operator)] = 1

                incorrect_bin_ops_vector = [
                    th.tensor(node_type_vectors[grand_parent]),
                    th.tensor(node_type_vectors[parent]),
                    th.tensor(other_operator_vector),
                    th.tensor(type_vectors[left_type]),
                    th.tensor(token_vectors[left]),
                    th.tensor(type_vectors[right_type]),
                    th.tensor(token_vectors[right]),
                ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)

                g = dgl.graph(binOps_graph, num_nodes=num_nodes)
                g.ndata['features'] = self.get_tensor_feature(incorrect_bin_ops_vector)
                self.graphs.append(g)
                self.labels.append(1)

            ## Wrong binary operand
            elif self.bug_type == 'incorrect_binary_operand':
                replace_left = random.random() < 0.5
                if replace_left:
                    to_replace_operand = left
                else:
                    to_replace_operand = right
                file = src.split(" : ")[0]
                all_operands = self.file_to_operands[file]
                tries_left = 100
                found = False
                while (not found) and tries_left > 0:
                    other_operand = random.choice(list(all_operands))
                    if other_operand.op in token_vectors and other_operand.op != to_replace_operand:
                        found = True
                    tries_left -= 1

                if not found:
                    return

                other_operand_vector = token_vectors[other_operand.op]
                other_operand_type_vector = type_vectors[other_operand.type]

                if replace_left:
                    incorrect_bin_operands_vector = [
                        th.tensor(node_type_vectors[grand_parent]),
                        th.tensor(node_type_vectors[parent]),
                        th.tensor(operator_vector),
                        th.tensor(other_operand_type_vector),
                        th.tensor(other_operand_vector),
                        th.tensor(type_vectors[right_type]),
                        th.tensor(token_vectors[right]),
                    ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)
                else:
                    incorrect_bin_operands_vector = [
                        th.tensor(node_type_vectors[grand_parent]),
                        th.tensor(node_type_vectors[parent]),
                        th.tensor(operator_vector),
                        th.tensor(type_vectors[left_type]),
                        th.tensor(token_vectors[left]),
                        th.tensor(other_operand_type_vector),
                        th.tensor(other_operand_vector),
                    ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)

                g = dgl.graph(binOps_graph, num_nodes=num_nodes)
                g.ndata['features'] = self.get_tensor_feature(incorrect_bin_operands_vector)
                self.graphs.append(g)
                self.labels.append(1)
        

    def generate_graphs_from_calls_ast(self):
        num_nodes = 8
        
        dataset = calls_training if self.is_training else calls_eval
        for call in dataset:
            arguments = call['arguments']
            if len(arguments) != 2:
                continue
            
            callee_string = call['callee']
            argument_strings = call['arguments']
            
            if not (callee_string in token_vectors):
                continue
            not_found = False
            for argument_string in argument_strings:
                if not (argument_string in token_vectors):
                    not_found = True
            if not_found:
                continue

            callee_vector = token_vectors[callee_string]
            argument0_vector = token_vectors[argument_strings[0]]
            argument1_vector = token_vectors[argument_strings[1]]
            
            base_string = call['base']
            base_vector = token_vectors.get(base_string, [0] * name_embedding_size)
            
            argument_type_strings = call['argumentTypes']
            argument0_type_vector = type_vectors.get(
                argument_type_strings[0], [0] * type_embedding_size)
            argument1_type_vector = type_vectors.get(
                argument_type_strings[1], [0] * type_embedding_size)
            
            parameter_strings = call['parameters']
            parameter0_vector = token_vectors.get(
                parameter_strings[0], [0] * name_embedding_size)
            parameter1_vector = token_vectors.get(
                parameter_strings[1], [0] * name_embedding_size)
            
            correct_vector = [
                th.tensor(base_vector),
                th.tensor(callee_vector),
                th.tensor(parameter0_vector),
                th.tensor(argument0_type_vector),
                th.tensor(argument0_vector),
                th.tensor(parameter1_vector),
                th.tensor(argument1_type_vector),
                th.tensor(argument1_vector),
            ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)
            
            g = dgl.graph(calls_graph, num_nodes=num_nodes)
            g.ndata['features'] = self.get_tensor_feature(correct_vector)
            self.graphs.append(g)
            self.labels.append(0)
            
            ## Swapped args
            swapped_args_vector = [
                th.tensor(base_vector),
                th.tensor(callee_vector),
                th.tensor(parameter0_vector),
                th.tensor(argument1_type_vector),
                th.tensor(argument1_vector),
                th.tensor(parameter1_vector),
                th.tensor(argument0_type_vector),
                th.tensor(argument0_vector),
            ] if self.use_deepbugs_embeddings else self.generate_random_embedding(num_nodes)

            g = dgl.graph(calls_graph, num_nodes=num_nodes)
            g.ndata['features'] = self.get_tensor_feature(swapped_args_vector)
            self.graphs.append(g)
            self.labels.append(1)
    
    @property
    def dataset_type(self):
        return 'training' if self.is_training else 'eval'

    def process(self):
        filepath = './data/large_graph_data_{}_{}_{}.bin'.format(
            self.dataset_type,
            'deepbugs' if self.use_deepbugs_embeddings else 'random',
            self.bug_type
        )
        if os.path.exists(filepath):
            print('----Loading {} graph data----'.format(self.dataset_type))
            self.graphs, label_dict = load_graphs(filepath)
            self.labels = label_dict['labels']
        else:
            print('----Saving {} graph data----'.format(self.dataset_type))
            if self.bug_type in ['incorrect_binary_operator', 'incorrect_binary_operand']:
                self.pre_scan_binOps(binOps_training, binOps_eval)
                self.generate_graphs_from_binOps_ast()
            elif self.bug_type == 'swapped_args':
                self.generate_graphs_from_calls_ast()

            random.shuffle(self.graphs)
            self.labels = th.LongTensor(self.labels)
            save_graphs(filepath, self.graphs, {'labels': self.labels})


    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]


    def __len__(self):
        return len(self.graphs)

    @property
    def num_classes(self):
        """Number of classes."""
        return 2


####################################################################################
### Create model for training and use the above dataset
import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

import dgl.function as fn
import torch
import torch.nn as nn


# Sends a message of node feature h.
msg = fn.copy_src(src='features', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'features': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['features'])
        h = self.activation(h)
        return {'features' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features.
        g.ndata['features'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('features')

import torch.nn.functional as F


class Classifier(nn.Module):
  def __init__(self, in_dim, hidden_dim, n_classes):
      super(Classifier, self).__init__()

      self.layers = nn.ModuleList([
          GCN(in_dim, hidden_dim, F.relu),
          GCN(hidden_dim, hidden_dim, F.relu),
          GCN(hidden_dim, hidden_dim, F.relu)])
      self.classify = nn.Linear(hidden_dim, n_classes)

  def forward(self, g):
      # For undirected graphs, in_degree is the same as
      # out_degree.
      h = g.ndata['features']
      for conv in self.layers:
          h = conv(g, h)
      g.ndata['features'] = h
      hg = dgl.mean_nodes(g, 'features')
      return self.classify(hg)

import torch.optim as optim
from torch.utils.data import DataLoader
# Create training and test sets.
trainset = CorrectAndBuggyDataset(use_deepbugs_embeddings=True, is_training=True, bug_type='incorrect_binary_operator')
testset = CorrectAndBuggyDataset(use_deepbugs_embeddings=True, is_training=False, bug_type='incorrect_binary_operator')

# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(trainset, batch_size=100, shuffle=True,
                         collate_fn=collate)

# Create model
model = Classifier(name_embedding_size, 256, trainset.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(20):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

## Evaluate model

model.eval()
# Convert a list of tuples to two lists
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

# With word2vec
# Accuracy of sampled predictions on the test set: 43.8165%
# Accuracy of argmax predictions on the test set: 47.616141%

# With random embeddings
# Accuracy of sampled predictions on the test set: 43.9176%
# Accuracy of argmax predictions on the test set: 47.616141%