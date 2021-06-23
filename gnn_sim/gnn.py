import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeWeightNorm, GraphConv, SAGEConv, GINConv, GATConv, DotGatConv

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler


#outcome
def brier(y_true, y_pred):
    return np.mean(np.sum(np.power(y_true - y_pred, 2), axis = 1))


sim_run = ''
sim_turn = 0
class ImageGraph(DGLDataset):
    def __init__(self):
        super().__init__(name = 'image')

    def process(self):
        folder = ''.join(['Code/Info/SimGraph/', sim_run, '/sim', str(sim_turn), '/', set_type])
        edges = pd.read_csv(''.join([folder, '/edges.csv']), index_col = 0)

        graphs = pd.read_csv(''.join([folder, '/graphs.csv']), index_col = 0)

        nodes = pd.read_csv(''.join([folder, '/nodes.csv']))
        

        self.graphs = []
        self.labels = []
        
        #create a graph for each graph ID from the edges table
        #first process 
        label_dict = {}
        num_nodes_dict = {}
        
        for _, row in graphs.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']
            
        #group edges/nodes by graph ID
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby('graph_id')
        
        #for each graph id
        for graph_id in edges_group.groups:
            #find the edges, nodes attributes, num_nodes and label
            edges_of_id = edges_group.get_group(graph_id)
            nodes_of_id = nodes_group.get_group(graph_id)
            
            src = edges_of_id['src'].to_numpy() - 1
            dst = edges_of_id['dst'].to_numpy() - 1
            weights = torch.from_numpy(edges_of_id['weight'].to_numpy())
            node_feature = nodes_of_id.loc[:, nodes.columns != 'graph_id'].to_numpy()
            node_feature = torch.from_numpy(node_feature)
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            
            #create a graph and add it to the list of graphs and labels
            g = dgl.graph((src, dst), num_nodes = num_nodes)
            g.ndata['f1'] = node_feature
            g = dgl.add_self_loop(g)
            add_weights = torch.ones(num_nodes)
            edge_weights = torch.cat((weights, add_weights))
            g.edata['w'] = edge_weights
            nodes_index = torch.nonzero(g.edges()[1] == (num_nodes-1))
            nodes_weights = edge_weights[nodes_index]
            g.ndata['w'] = nodes_weights.float()
            lw = np.concatenate((np.zeros(num_nodes-1), np.ones(1))).reshape((num_nodes, 1))
            g.ndata['lw'] = torch.from_numpy(lw).float()
            
            self.graphs.append(g)
            self.labels.append(label)
        
        #convert the label list to tensor for saving
        self.labels = torch.LongTensor(self.labels)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, p, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats, norm='none', weight=True, bias = True)
        self.conv2 = GraphConv(h1_feats, h2_feats, norm='none', weight=True, bias = True)
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        #self.dropout3 = nn.Dropout(p)
        self.dense1 = nn.Linear(h2_feats, h3_feats)
        self.classify = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout2(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h', 'lw')
        out1 = self.dense1(hg)
        out1 = F.relu(out1)
        out1 = self.dropout3(out1)
        return self.classify(out1)  
    
class SGCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h3_feats, p, num_classes):
        super(SGCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, bias = True, feat_drop = p, aggregator_type = 'gcn')
        self.conv2 = SAGEConv(h1_feats, h3_feats, bias = True, feat_drop = p, aggregator_type = 'gcn')
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        #self.dropout3 = nn.Dropout(p)
        self.classify = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
    
class GIN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, p, num_classes):
        super(GIN, self).__init__()
        self.fun1 = nn.Linear(in_feats, h1_feats)
        self.fun2 = nn.Linear(h1_feats, h2_feats)
        self.conv1 = GINConv(self.fun1, 'sum')
        self.conv2 = GINConv(self.fun2, 'sum')
        
        self.BN1 = nn.BatchNorm1d(h1_feats)
        self.BN2 = nn.BatchNorm1d(h2_feats)
        self.BN3 = nn.BatchNorm1d(h3_feats)
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        #self.dropout3 = nn.Dropout(p)
        self.dense1 = nn.Linear(h2_feats, h3_feats)
        self.classify = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout2(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h', 'lw')
        out1 = self.dense1(hg)
        out1 = F.relu(out1)
        out1 = self.dropout3(out1)
        return self.classify(out1)   


#calculate accuray given model and dataset
def get_accu(dataloader, model):
    
    num_correct = 0
    num_tests = 0
    
    sum_loss = 0
    
    auc_pred = []
    auc_label = []
    
    for batched_graph, labels in dataloader:
        pred = model(batched_graph, 
                     batched_graph.ndata['f1'].float(),
                         batched_graph.edata['w'].float())
        correct = (pred.argmax(1) == labels).sum().item()
        num_correct += correct
        sum_loss += F.cross_entropy(pred, labels, reduction = 'sum')
        num_tests += len(labels)
        
        auc_pred = auc_pred + list(pred.detach().numpy()[:,1].reshape(-1))
        auc_label= auc_label + list((labels.numpy()).reshape(-1))
    
    return num_correct / num_tests, sum_loss.detach().numpy() / num_tests, metrics.roc_auc_score(auc_label, auc_pred)

def get_ba(dataloader, model):
    
    for batched_graph, labels in dataloader:
        preds = model(batched_graph, 
                     batched_graph.ndata['f1'].float(),
                         batched_graph.edata['w'].float())       
        soft = nn.Softmax(dim = 1)
        preds = soft(preds)
        preds = preds.detach().numpy()
        trues = F.one_hot(labels, num_classes = 2)
        trues = trues.detach().numpy()
        brier_score = brier(trues, preds)
    return brier_score
        
#train model, including calculating accuracy
def train_model(model, epochs = 100, lr = 0.001):
    
    train_accu = []
    test_accu = []
    
    train_loss = []
    test_loss = []
    
    test_brier = []
    test_auc = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0)
    for epoch in range(epochs):
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, 
                     batched_graph.ndata['f1'].float(),
                         batched_graph.edata['w'].float())
            loss = F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_one_accu, train_one_loss, train_one_auc = get_accu(train_dataloader, model)
        test_one_accu, test_one_loss, test_one_auc = get_accu(test_dataloader, model)
        test_one_brier = get_ba(test_dataloader, model)
        train_accu.append(train_one_accu)
        test_accu.append(test_one_accu)
        train_loss.append(train_one_loss)
        test_loss.append(test_one_loss)
        test_brier.append(test_one_brier)
        test_auc.append(test_one_auc)
        print([epoch, test_one_accu])
    return model, train_accu, test_accu, train_loss, test_loss, test_brier, test_auc


sims = [2, 3, 4, 7]
for num_sim in sims:
    for sim_turn in range(1, 101):
    
        print(num_sim)
        sim_run = ''.join(['graphSim', str(num_sim)])
        set_type = 'train'
        train_graph = ImageGraph()
        set_type = 'test'
        test_graph = ImageGraph()  
           
        #get data loader
        train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
        test_sampler = SubsetRandomSampler(torch.arange(len(test_graph)))
        
        train_dataloader = GraphDataLoader(train_graph, batch_size = 32, 
                                           sampler = train_sampler, drop_last = False)
        test_dataloader = GraphDataLoader(test_graph, batch_size = len(test_graph), 
                                          sampler = test_sampler, drop_last = False)
        gnn_model = GIN(128, 256, 128, 128, 0.2, 2)
        
        _, gnn_train_accu, gnn_test_accu, gnn_train_loss, gnn_test_loss, gnn_test_brier, gnn_test_auc = train_model(gnn_model)
        
        gnn_results = pd.DataFrame({'train_accu': gnn_train_accu,
                                    'train_loss': gnn_train_loss,
                                    'test_accu': gnn_test_accu,
                                    'test_loss': gnn_test_loss,
                                    'test_brier':gnn_test_brier,
                                    'test_auc':gnn_test_auc})
        
        
        hist_csv_file = ''.join(['Code/Info/SimGraph/graphSim', str(num_sim), '/sim', str(sim_turn), '/gnn_results.csv'])
        with open(hist_csv_file, mode='w') as f:
            gnn_results.to_csv(f)
            




    



        
        
        