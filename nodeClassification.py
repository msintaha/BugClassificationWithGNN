
# Contruct a two-layer GNN model
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

import dgl
import torch.optim as optim
from torch.utils.data import DataLoader

def main(bug_type, use_deepbugs_embeddings, dataset_size):
    print('----Node Classification Training on Mono graphs in bug type {} with {}----'.format(bug_type, 'deepbugs embeddings' if use_deepbugs_embeddings else 'random embeddings'))
    # Create training and test sets.
    
    from labeledNodes_mini_dataset import MiniCorrectAndBuggyDataset
    trainset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=True, bug_type=bug_type)
    testset = MiniCorrectAndBuggyDataset(use_deepbugs_embeddings=use_deepbugs_embeddings, is_training=False, bug_type=bug_type)


    print (trainset[0])

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=100, shuffle=True,
                            collate_fn=collate)
    
    # node_features = graph.ndata['features']
    # node_labels = graph.ndata['nodeLabels']
    # train_mask = graph.ndata['train_mask']
    # valid_mask = graph.ndata['val_mask']
    # test_mask = graph.ndata['test_mask']
    n_features = 200
    n_labels = 2

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
    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
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
