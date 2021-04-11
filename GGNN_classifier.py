import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.nn.pytorch import *
from torch.utils.data import DataLoader


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


"""
Gated Graph Neural Network module for graph classification tasks
"""
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
import torch
from torch import nn


class GraphClsGGNN(nn.Module):
    def __init__(self,
                 annotation_size,
                 out_feats,
                 n_steps,
                 n_etypes,
                 num_cls):
        super(GraphClsGGNN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats

        self.ggnn = GatedGraphConv(in_feats=out_feats,
                                   out_feats=out_feats,
                                   n_steps=n_steps,
                                   n_etypes=n_etypes)

        pooling_gate_nn = nn.Linear(annotation_size + out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn)
        self.output_layer = nn.Linear(annotation_size + out_feats, num_cls)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, graph, labels=None):
        etypes = graph.edata.pop('type')
        annotation = graph.ndata.pop('annotation').float()

        assert annotation.size()[-1] == self.annotation_size

        node_num = graph.number_of_nodes()

        zero_pad = torch.zeros([node_num, self.out_feats - self.annotation_size],
                               dtype=torch.float,
                               device=annotation.device)

        h1 = torch.cat([annotation, zero_pad], -1)
        out = self.ggnn(graph, h1, etypes)

        out = torch.cat([out, annotation], -1)

        out = self.pooling(graph, out)

        logits = self.output_layer(out)
        preds = torch.argmax(logits, -1)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, preds
        return preds


def main(bug_type, use_deepbugs_embeddings, dataset_size):
    print('----GGNN Classifier Training bug type {} with {}----'.format(bug_type, 'deepbugs embeddings' if use_deepbugs_embeddings else 'random embeddings'))
    # Create training and test sets.
    if dataset_size == 'mini':
        from homogenous_mini_dataset import MiniCorrectAndBuggyDataset
        trainset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
        testset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)
    elif dataset_size == 'full':
        from homogenous_full_dataset import FullCorrectAndBuggyDataset
        trainset = FullCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
        testset = FullCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)

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
    model = GraphClsGGNN(annotation_size=2,
                         out_feats=2,
                         n_steps=5,
                         n_etypes=1,
                         num_cls=2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


    epoch_losses = []
    for epoch in range(20):
        model.train()
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
