import dgl


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sends a message of node feature h.
msg = fn.copy_src(src='features', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'features': accum}



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['features']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['features'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'features', ntype=ntype)
            return self.classify(hg)

import torch.optim as optim
from torch.utils.data import DataLoader

def main(bug_type, use_deepbugs_embeddings, dataset_size):
    print('----RGCN Training on hetero graphs in bug type {} with {}----'.format(bug_type, 'deepbugs embeddings' if use_deepbugs_embeddings else 'random embeddings'))
    # Create training and test sets.
    if dataset_size == 'mini':
        from heterogenous_mini_dataset import MiniCorrectAndBuggyDataset
        trainset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
        testset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)
    elif dataset_size == 'full':
        from heterogenous_full_dataset import FullCorrectAndBuggyDataset
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
        prediction = model(test_bg)
        probs_Y = torch.softmax(prediction, 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Create model
    model = HeteroClassifier(200, 16, trainset.num_classes, \
      ['follows', 'precedes', 'precedes', 'precedes'] if bug_type == 'swapped_args' else ['parent', 'parent', 'precedes', 'precedes', 'followed_by', 'follows','binOpTypes','typeOf'])
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.train()

    epoch_losses = []
    for epoch in range(30):
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
        if epoch % 5 == 0:
          evaluate()

    evaluate()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bug_type', help='Type of bug to train', choices=['swapped_args', 'binOps', 'incorrect_binary_operator', 'incorrect_binary_operand'], required=False)
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
