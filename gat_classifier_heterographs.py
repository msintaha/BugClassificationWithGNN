import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout))
        for l in range(1, num_heads):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads,
                                        hidden_size, num_heads, dropout))
        self.predict = nn.Linear(hidden_size * num_heads, out_size)

    def forward(self, g):
        h = g.ndata['features']
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

import torch.optim as optim
from torch.utils.data import DataLoader

def main(bug_type, use_deepbugs_embeddings, dataset_size):
    print('----GATConv Training on hetero graphs in bug type {} with {}----'.format(bug_type, 'deepbugs embeddings' if use_deepbugs_embeddings else 'random embeddings'))
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
        probs_Y = torch.softmax(model(test_bg), 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    # Create model
    # model = HeteroClassifier(200, 16, trainset.num_classes, 8, ['precedes', 'precedes', 'precedes', 'follows', 'follows', 'precedes'])
    model = HAN(meta_paths=[['follows', 'precedes']],
                in_size=200,
                hidden_size=16,
                out_size=trainset.num_classes,
                num_heads=8,
                dropout=0.6)
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
    '--bug_type', help='Type of bug to train', choices=['swapped_args', 'binOps'], required=False)
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
