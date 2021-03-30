import dgl
from homogenous_mini_dataset import MiniCorrectAndBuggyDataset
from homogenous_full_dataset import FullCorrectAndBuggyDataset


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn import GATConv

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
# from torch_metrics import Accuracy, Precision, Recall

def main(bug_type, use_deepbugs_embeddings, dataset_size):
    print('----GCN Classifier Training bug type {} with {}----'.format(bug_type, 'deepbugs embeddings' if use_deepbugs_embeddings else 'random embeddings'))
    # Create training and test sets.
    if dataset_size == 'mini':
        trainset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
        testset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)
    elif dataset_size == 'full':
        trainset = FullCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
        testset = FullCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=100, shuffle=True,
                            collate_fn=collate)
    
    def evaluate():
        ## Evaluate model
        model.eval()
        # Convert a list of tuples to two lists
        test_X, test_Y = map(list, zip(*testset))
        test_bg = dgl.batch(test_X)

        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        model_output = model(test_bg)
        probs_Y = torch.softmax(model_output, 1)
        # print('probs_Y', probs_Y)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))


    # Create model
    model = Classifier(200, 256, trainset.num_classes)
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
        if epoch % 5 == 0:
            evaluate()
        epoch_losses.append(epoch_loss)

    evaluate()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bug_type', help='Type of bug to train', choices=['swapped_args', 'incorrect_binary_operator', 'incorrect_binary_operand', 'all'], required=False)
parser.add_argument(
    '--use_deepbugs_embeddings', help='Random or deepbugs embeddings', required=False)
parser.add_argument(
    '--dataset_size', help='Mini or Full dataset', choices=['mini', 'full'], required=False)


if __name__=='__main__': 
    args = parser.parse_args()
    bug_type = args.bug_type or 'all'
    use_deepbugs_embeddings = True if args.use_deepbugs_embeddings in ['True', 'true'] else False
    dataset_size = args.dataset_size or 'mini'
    main(bug_type, use_deepbugs_embeddings, dataset_size) 


# With word2vec
# Accuracy of sampled predictions on the test set: 43.8165%
# Accuracy of argmax predictions on the test set: 47.616141%

# With random embeddings
# Accuracy of sampled predictions on the test set: 43.9176%
# Accuracy of argmax predictions on the test set: 47.616141%