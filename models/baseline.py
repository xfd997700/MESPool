# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:01:38 2022

@author: Fanding Xu
"""

import math
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, BatchNorm2d, ReLU, LeakyReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch_geometric.nn import (GINConv, GCNConv, GATConv, SAGEConv,
                                DenseGINConv, DenseGCNConv,
                                AttentiveFP, MLP,
                                TopKPooling, SAGPooling, dense_diff_pool, dense_mincut_pool, EdgePooling, ASAPooling,
                                GraphMultisetTransformer, global_add_pool,
                                BatchNorm)
from layers import EGIN, DenseEGIN, HaarPooling, Uext_batch
from layers.edgepool_e import EdgePooling_E
# from layers.mincutpool import dense_mincut_pool
from layers.utils import reset, generate_edge_batch
from torch_scatter import scatter_add
import time
class GNNPred(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_classes,
        num_layers = 3,
        dropout = 0.2,
        ratio = 0.5,
        lin_before_conv = True,
        **kwargs):
        super().__init__()
        self.lin_before_conv = lin_before_conv
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.bns_embd = ModuleList()
        if 'max_nodes' in kwargs:
            kwargs.pop('max_nodes')
        
        
        if lin_before_conv:
            self.lin_start = Linear(in_channels, hidden_channels)
            self.bn_start = torch.nn.BatchNorm1d(hidden_channels)
            in_channels = hidden_channels   
        
        
        for i in range(num_layers-1):
            conv = self.init_conv(in_channels, hidden_channels, **kwargs)
            self.convs.append(conv)
            in_channels = hidden_channels
            
        for i in range(num_layers-1):
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels))
            self.bns_embd.append(
                torch.nn.BatchNorm1d(hidden_channels))
            
        conv = self.init_conv(hidden_channels, out_channels, **kwargs)    
        self.convs.append(conv)
        

        self.bns.append(
            torch.nn.BatchNorm1d(out_channels))
        self.bns_embd.append(
            torch.nn.BatchNorm1d(out_channels))
        
        if self.if_pool:
            self.pools = ModuleList()
            for i in range(num_layers):
                self.pools.append(
                    self.init_pool(hidden_channels, ratio=ratio, **kwargs))
        
        self.lin = Linear(hidden_channels*(num_layers-1)+out_channels, num_classes)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.bns:
            norm.reset_parameters()
        for norm in self.bns_embd:
            norm.reset_parameters()
        if self.if_pool:
            for pool in self.pools:
                pool.reset_parameters()
        self.lin.reset_parameters()

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        raise NotImplementedError
        
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        raise NotImplementedError
    
    def step_pool(self, pool, x, edge_index, batch):
        return x, edge_index, batch
    
    @property
    def if_pool(self):
        return False
    
    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x = data.x
        edge_index = data.edge_index
        embds = []
        
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
            
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index)

            x = torch.relu(self.bns[i](x))
            if self.if_pool:
                x, edge_index, batch = self.step_pool(self.pools[i], x, edge_index, batch)
            embd = global_add_pool(x, batch)
            embd = F.relu(self.bns_embd[i](embd))
            if self.dropout is not None:
                embd = F.dropout(embd, p=self.dropout, training=self.training)
            embds.append(embd)   
        embds = torch.cat(embds, dim=-1)
        return embds
    
    def forward(self, data):
        self.perm = []
        self.score = []
        self.x = []
        embds = self.GNN(data)
        out = self.lin(embds)
        return out
    
    

class GCN(GNNPred):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(GNNPred):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(GNNPred):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        mlp = Sequential(Linear(in_channels, out_channels),
                           BatchNorm1d(out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels))
        return GINConv(mlp, **kwargs)

class GAT(GNNPred):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        return GATConv(in_channels, out_channels, **kwargs)


class TopK(GIN):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return TopKPooling(in_channels=in_channels, ratio=ratio, **kwargs)
    
    def step_pool(self, pool, x, edge_index, batch):
        self.x.append(x)
        x, edge_index, _, batch, perm, score = pool(x=x, edge_index=edge_index, batch=batch)
        self.perm.append(perm)
        self.score.append(score)
        return x, edge_index, batch

class SAG(GIN):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return SAGPooling(in_channels=in_channels, ratio=ratio, **kwargs)
    
    def step_pool(self, pool, x, edge_index, batch):
        self.x.append(x)
        x, edge_index, _, batch, perm, score = pool(x=x, edge_index=edge_index, batch=batch)
        self.perm.append(perm)
        self.score.append(score)
        return x, edge_index, batch
    
class ASAP(GIN):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return ASAPooling(in_channels=in_channels, ratio=ratio, **kwargs)
    
    def step_pool(self, pool, x, edge_index, batch):
        x, edge_index, _, batch, _ = pool(x=x, edge_index=edge_index, batch=batch)
        return x, edge_index, batch


class EdgePool(GIN):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio=None,
                  **kwargs):
        return EdgePooling(in_channels=in_channels, **kwargs)
    
    def step_pool(self, pool, x, edge_index, batch):
        x, edge_index, batch, _ = pool(x=x, edge_index=edge_index, batch=batch)
        return x, edge_index, batch


class GMT(GIN):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_classes,
                 num_layers = 3,
                 dropout = 0.2,
                 ratio = 0.5,
                 lin_before_conv = True,
                 **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_classes, num_layers, dropout,ratio, lin_before_conv, **kwargs)        
        self.global_pool = GraphMultisetTransformer(out_channels, out_channels, out_channels, pooling_ratio=ratio)
        self.lin = Linear(out_channels, num_classes)
        self.reset_parameters_gmt()
        
    def reset_parameters_gmt(self):
        self.global_pool.reset_parameters()
        self.lin.reset_parameters()
        
    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x = data.x
        edge_index = data.edge_index
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
        
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = F.relu(self.bns[i](x))
            if self.if_pool and i < self.num_layers-1:
                x, edge_index, batch = self.step_pool(self.pools[i], x, edge_index, batch)
        embd = self.global_pool(x=x, edge_index=edge_index, batch=batch)
        embd = F.relu(self.bns_embd[i](embd))
        if self.dropout is not None:
            embd = F.dropout(embd, p=self.dropout, training=self.training)
        
        return embd


class DensePool(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_classes,
        num_layers = 3,
        dropout = 0.2,
        ratio = 0.5,
        lin_before_conv = True,
        max_nodes = 150,
        **kwargs):
        super().__init__()
        self.lin_before_conv = lin_before_conv
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.convs_pool = ModuleList()
        self.bns = ModuleList()
        self.bnsc = ModuleList()
        self.bns_embd = ModuleList()

        self.max_nodes = max_nodes
        if lin_before_conv:
            self.lin_start = Linear(in_channels, hidden_channels)
            self.bn_start = torch.nn.BatchNorm1d(hidden_channels)
            in_channels = hidden_channels   
        
        num_nodes = max_nodes
        for i in range(num_layers-1):
            num_nodes = math.ceil(ratio * num_nodes)
            conv = self.init_conv(in_channels, hidden_channels, **kwargs)
            self.convs.append(conv)
            conv_pool = self.init_conv_pool(in_channels, num_nodes, **kwargs)
            self.convs_pool.append(conv_pool)
            in_channels = hidden_channels
            
        for i in range(num_layers-1):
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels))
            self.bnsc.append(
                torch.nn.BatchNorm1d(hidden_channels))
            self.bns_embd.append(
                torch.nn.BatchNorm1d(hidden_channels))
            
        conv = self.init_conv(hidden_channels, out_channels, **kwargs)    
        self.convs.append(conv)
        self.convs_pool.append(self.init_conv_pool(hidden_channels, math.ceil(ratio * num_nodes), **kwargs))
    
        self.bns.append(
            torch.nn.BatchNorm1d(out_channels))
        self.bnsc.append(
            torch.nn.BatchNorm1d(out_channels))
        self.bns_embd.append(
            torch.nn.BatchNorm1d(out_channels))
        
        self.lin = Linear(hidden_channels*(num_layers-1)+out_channels, num_classes)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
            
        # for conv_pool in self.convs_pool:
        #     conv_pool.reset_parameters()
        for norm in self.bnsc:
            norm.reset_parameters()    

        for norm in self.bns:
            norm.reset_parameters()
        for norm in self.bns_embd:
            norm.reset_parameters()
        self.lin.reset_parameters()
        
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        mlp = Sequential(Linear(in_channels, out_channels),
                         ReLU(),
                         Linear(out_channels, out_channels),)
                         # ReLU())
        return DenseGINConv(mlp, **kwargs)
    
    def init_conv_pool(self, in_channels: int, out_channels: int,
                       **kwargs):
        raise NotImplementedError

    def densepool(self, x, adj, s, mask=None,
                  **kwargs):
        raise NotImplementedError
        
    def cal_s(self, i, x, adj, mask=None):
        raise NotImplementedError

    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x, mask = to_dense_batch(data.x, batch, max_num_nodes=self.max_nodes)
        # x = data.x
        adj = to_dense_adj(data.edge_index, batch, max_num_nodes=self.max_nodes)
        embds = []
        
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
        
        pool_loss = 0
        
        for i in range(self.num_layers):
            s = self.cal_s(i, x, adj, mask)
            x = self.convs[i](x, adj, mask)
            x = self.bnsc[i](x.permute(0,2,1)).permute(0,2,1).relu()
            
            x, adj, l = self.densepool(x, adj, s)
            pool_loss = pool_loss + l
            # x = self.bns[i](x.permute(0,2,1)).permute(0,2,1)
            embd = x.sum(dim = 1)
            embd = F.relu(self.bns_embd[i](embd))
            if self.dropout is not None:
                embd = F.dropout(embd, p=self.dropout, training=self.training)
            embds.append(embd)   
            mask = None
        embds = torch.cat(embds, dim=-1)
        
        return embds, pool_loss

    def forward(self, data):
        self.perm = []
        self.score = []
        self.x = []
        embds, pool_loss = self.GNN(data)
        out = self.lin(embds)
        # print(embds[0])
        return out, pool_loss

class Diff(DensePool):
    def init_conv_pool(self, in_channels: int, out_channels: int,
                       **kwargs):
        mlp = Sequential(Linear(in_channels, out_channels),
                         ReLU(),
                         Linear(out_channels, out_channels),
                         ReLU())
        return DenseGINConv(mlp, **kwargs)
    
    def densepool(self, x, adj, s, mask=None, lam = 0.5,
                  **kwargs):
        x, adj, ll, el = dense_diff_pool(x, adj, s)
        l = lam * ll + (1-lam) * el
        return x, adj, l
    
    def cal_s(self, i, x, adj, mask=None):
        return self.convs_pool[i](x, adj, mask)

class MinCut(DensePool):
    def init_conv_pool(self, in_channels: int, out_channels: int,
                       **kwargs):
        mlp = Sequential(Linear(in_channels, out_channels),
                         ReLU(),
                         Linear(out_channels, out_channels),
                         ReLU())
        return mlp
    
    def densepool(self, x, adj, s, mask=None, lam = 0.5,
                  **kwargs):
        x, adj, ml, ol = dense_mincut_pool(x, adj, s)
        l = lam * ml + (1-lam) * ol
        return x, adj, l
    
    def cal_s(self, i, x, adj, mask=None):
        return self.convs_pool[i](x)

class GNNPred_e(torch.nn.Module):
    def __init__(
        self,
        in_channels_x,
        hidden_channels_x,
        out_channels_x,
        in_channels_e,
        hidden_channels_e,
        out_channels_e,
        num_classes,
        num_layers = 3,
        dropout = 0.2,
        ratio = 0.5,
        lin_before_conv = True,
        **kwargs):
        super().__init__()
        self.lin_before_conv = lin_before_conv
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.bns_embd = ModuleList()
        if 'max_nodes' in kwargs:
            kwargs.pop('max_nodes')
        
        
        if lin_before_conv:
            self.lin_start = Linear(in_channels_x, hidden_channels_x)
            self.bn_start = torch.nn.BatchNorm1d(hidden_channels_x)
            in_channels_x = hidden_channels_x  
        
        
        for i in range(num_layers-1):
            conv = self.init_conv(in_channels_x, in_channels_e, hidden_channels_x, **kwargs)
            self.convs.append(conv)
            in_channels_x = hidden_channels_x
            in_channels_e = hidden_channels_e
            
        for i in range(num_layers-1):
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels_x))
            self.bns_embd.append(
                torch.nn.BatchNorm1d(hidden_channels_x))
            
        conv = self.init_conv(hidden_channels_x, hidden_channels_e, out_channels_x, **kwargs)    
        self.convs.append(conv)
        

        self.bns.append(
            torch.nn.BatchNorm1d(out_channels_x))
        self.bns_embd.append(
            torch.nn.BatchNorm1d(out_channels_x))
        
        if self.if_pool:
            self.pools = ModuleList()
            for i in range(num_layers):
                self.pools.append(
                    self.init_pool(hidden_channels_x, ratio=ratio, **kwargs))
        
        self.lin = Linear(hidden_channels_x*(num_layers-1)+out_channels_x, num_classes)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.bns:
            norm.reset_parameters()
        for norm in self.bns_embd:
            norm.reset_parameters()
        if self.if_pool:
            for pool in self.pools:
                pool.reset_parameters()
        self.lin.reset_parameters()

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        raise NotImplementedError
        
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        raise NotImplementedError
    
    def step_pool(self, pool, x, edge_index, batch):
        return x, edge_index, batch
    
    @property
    def if_pool(self):
        return False
    
    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        embds = []
        
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
            
        for i in range(self.num_layers):
            x, edge_attr = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)

            # x = torch.relu(self.bns[i](x))
            if self.if_pool:
                x, edge_index, edge_attr, batch = self.step_pool(self.pools[i], x, edge_index, edge_attr, batch)
            embd = global_add_pool(x, batch)
            embd = F.relu(self.bns_embd[i](embd))
            if self.dropout is not None:
                embd = F.dropout(embd, p=self.dropout, training=self.training)
            embds.append(embd)   
        embds = torch.cat(embds, dim=-1)
        return embds
    
    def forward(self, data):
        self.perm = []
        self.score = []
        self.x = []
        embds = self.GNN(data)
        out = self.lin(embds)
        return out


class GIN_e(GNNPred_e):
    def init_conv(self, in_channels_x, in_channels_e, hidden_channels_x,
                  **kwargs):
        return EGIN(in_channels_x, in_channels_e, hidden_channels_x, **kwargs)


class TopK_e(GIN_e):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return TopKPooling(in_channels=in_channels, ratio=ratio, **kwargs)
    
    def step_pool(self, pool, x, edge_index, edge_attr, batch):
        self.x.append(x)
        x, edge_index, edge_attr, batch, perm, score = pool(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        self.perm.append(perm)
        self.score.append(score)
        return x, edge_index, edge_attr, batch

class SAG_e(GIN_e):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return SAGPooling(in_channels=in_channels, ratio=ratio, **kwargs)
    
    def step_pool(self, pool, x, edge_index, edge_attr, batch):
        self.x.append(x)
        x, edge_index, edge_attr, batch, perm, score = pool(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        return x, edge_index, edge_attr, batch



class EdgePool_e(GIN_e):
    @property
    def if_pool(self):
        return True
    
    def init_pool(self, in_channels: int, ratio: float,
                  **kwargs):
        return EdgePooling_E(in_channels_x=in_channels, in_channels_e=in_channels, **kwargs)
    
    def step_pool(self, pool, x, edge_index, edge_attr, batch):
        self.x.append(x)
        x, edge_index, edge_attr, batch, unpool_info = pool(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        return x, edge_index, edge_attr, batch






class DensePool_e(torch.nn.Module):
    def __init__(
        self,
        in_channels_x,
        hidden_channels_x,
        out_channels_x,
        in_channels_e,
        hidden_channels_e,
        out_channels_e,
        num_classes,
        num_layers = 3,
        dropout = 0.2,
        ratio = 0.5,
        lin_before_conv = True,
        max_nodes = 150,
        **kwargs):
        super().__init__()
        self.lin_before_conv = lin_before_conv
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.convs_pool = ModuleList()
        self.bns = ModuleList()
        self.bnsc = ModuleList()
        self.bns_embd = ModuleList()

        self.max_nodes = max_nodes
        if lin_before_conv:
            self.lin_start = Linear(in_channels_x, hidden_channels_x)
            self.bn_start = torch.nn.BatchNorm1d(hidden_channels_x)
            in_channels_x = hidden_channels_x  
        
        
        num_nodes = max_nodes
        for i in range(num_layers-1):
            num_nodes = math.ceil(ratio * num_nodes)
            conv = self.init_conv(in_channels_x, in_channels_e, hidden_channels_x, **kwargs)
            self.convs.append(conv)
            conv_pool = self.init_conv_pool(in_channels_x, num_nodes, **kwargs)
            self.convs_pool.append(conv_pool)
            in_channels_x = hidden_channels_x
            in_channels_e = hidden_channels_e
            
        for i in range(num_layers-1):
            self.bnsc.append(
                torch.nn.BatchNorm1d(hidden_channels_x))
            self.bns_embd.append(
                torch.nn.BatchNorm1d(hidden_channels_x))
            
        conv = self.init_conv(hidden_channels_x, hidden_channels_e, out_channels_x, **kwargs)    
        self.convs.append(conv)
        self.convs_pool.append(self.init_conv_pool(hidden_channels_x, math.ceil(ratio * num_nodes), **kwargs))
    

        self.bnsc.append(
            torch.nn.BatchNorm1d(out_channels_x))
        self.bns_embd.append(
            torch.nn.BatchNorm1d(out_channels_x))
        
        self.lin = Linear(hidden_channels_x*(num_layers-1)+out_channels_x, num_classes)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
            
        # for conv_pool in self.convs_pool:
        #     conv_pool.reset_parameters()
        for norm in self.bnsc:
            norm.reset_parameters()    

        for norm in self.bns_embd:
            norm.reset_parameters()
        self.lin.reset_parameters()
        
    def init_conv(self, in_channels_x, in_channels_e, hidden_channels_x,
                  **kwargs):
        return DenseEGIN(in_channels_x, in_channels_e, hidden_channels_x, **kwargs)
    
    def init_conv_pool(self, in_channels_x, num_nodes,
                       **kwargs):
        raise NotImplementedError

    def densepool(self, x, adj, s, mask=None,
                  **kwargs):
        raise NotImplementedError
        
    def cal_s(self, i, x, adj, mask=None):
        raise NotImplementedError

    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        x, mask = to_dense_batch(data.x, batch, max_num_nodes=self.max_nodes)
        # x = data.x
        adj = to_dense_adj(data.edge_index, batch, max_num_nodes=self.max_nodes)
        embds = []
        B, N, _ = adj.size()
        edge_attr = data.edge_attr
        e = x.new_zeros([B, N, N, edge_attr.size(1)])
        e[adj.to(torch.bool)] = edge_attr
        
        
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
        
        pool_loss = 0
        
        for i in range(self.num_layers):
            s = self.cal_s(i, x, adj, mask)
            x, e = self.convs[i](x, adj, e, mask)
            x = self.bnsc[i](x.permute(0,2,1)).permute(0,2,1).relu()
            
            x, adj, e, l = self.densepool(x, adj, e, s)
            pool_loss = pool_loss + l
            # x = self.bns[i](x.permute(0,2,1)).permute(0,2,1)
            embd = x.sum(dim = 1)
            embd = F.relu(self.bns_embd[i](embd))
            if self.dropout is not None:
                embd = F.dropout(embd, p=self.dropout, training=self.training)
            embds.append(embd)   
            mask = None
        embds = torch.cat(embds, dim=-1)
        
        return embds, pool_loss

    def forward(self, data):
        self.perm = []
        self.score = []
        self.x = []
        embds, pool_loss = self.GNN(data)
        out = self.lin(embds)
        # print(embds[0])
        return out, pool_loss


class Diff_e(DensePool_e):
    def init_conv_pool(self, in_channels_x, num_nodes,
                       **kwargs):
        mlp = Sequential(Linear(in_channels_x, num_nodes),
                         ReLU(),
                         Linear(num_nodes, num_nodes),
                         ReLU())
        return DenseGINConv(mlp, **kwargs)
    
    def densepool(self, x, adj, e, s, mask=None, lam = 0.5,
                  **kwargs):
        x, adj, e, ll, el = edge_diff_pool(x, adj, e, s)
        l = lam * ll + (1-lam) * el
        return x, adj, e, l
    
    def cal_s(self, i, x, adj, mask=None):
        return self.convs_pool[i](x, adj, mask)

class MinCut_e(DensePool_e):
    def init_conv_pool(self, in_channels_x, num_nodes,
                       **kwargs):
        mlp = Sequential(Linear(in_channels_x, num_nodes),
                         ReLU(),
                         Linear(num_nodes, num_nodes),
                         ReLU())
        return mlp
    
    def densepool(self, x, adj, e, s, mask=None, lam = 0.5,
                  **kwargs):
        x, adj, e, ml, ol = edge_mincut_pool(x, adj, e, s)
        l = lam * ml + (1-lam) * ol
        return x, adj, e, l
    
    def cal_s(self, i, x, adj, mask=None):
        return self.convs_pool[i](x)




def edge_diff_pool(x, adj, e, s):
    EPS = 1e-15
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    e = e.unsqueeze(0) if e.dim() == 3 else e
    
    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    out_e = torch.einsum('bij,bjle,blk->bike', s.transpose(1, 2), e, s)
    
    
    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, out_e, link_loss, ent_loss


def edge_mincut_pool(x, adj, e, s):
    EPS = 1e-15
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    e = e.unsqueeze(0) if e.dim() == 3 else e
    
    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    out_e = torch.einsum('bij,bjle,blk->bike', s.transpose(1, 2), e, s)
    
    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, out_e, mincut_loss, ortho_loss


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out




class HaarPool(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_classes,
        num_layers = 3,
        dropout = 0.2,
        ratio = 0.5,
        lin_before_conv = True,
        **kwargs):
        super().__init__()
        self.lin_before_conv = lin_before_conv
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.bns_embd = ModuleList()
        if 'max_nodes' in kwargs:
            kwargs.pop('max_nodes')
        
        
        if lin_before_conv:
            self.lin_start = Linear(in_channels, hidden_channels)
            self.bn_start = torch.nn.BatchNorm1d(hidden_channels)
            in_channels = hidden_channels   
        
        
        for i in range(num_layers-1):
            conv = self.init_conv(in_channels, hidden_channels, **kwargs)
            self.convs.append(conv)
            in_channels = hidden_channels
            
        for i in range(num_layers-1):
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels))
            self.bns_embd.append(
                torch.nn.BatchNorm1d(hidden_channels))
            
        conv = self.init_conv(hidden_channels, out_channels, **kwargs)    
        self.convs.append(conv)
        

        self.bns.append(
            torch.nn.BatchNorm1d(out_channels))
        self.bns_embd.append(
            torch.nn.BatchNorm1d(out_channels))
        
        
        self.pools = ModuleList()
        for i in range(num_layers):
            self.pools.append(HaarPooling(i+1))
        
        self.lin = Linear(hidden_channels*(num_layers-1)+out_channels, num_classes)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.lin_before_conv:
            self.lin_start.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.bns:
            norm.reset_parameters()
        for norm in self.bns_embd:
            norm.reset_parameters()
        if self.if_pool:
            for pool in self.pools:
                pool.reset_parameters()
        self.lin.reset_parameters()

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        mlp = Sequential(Linear(in_channels, out_channels),
                            BatchNorm1d(out_channels),
                            ReLU(),
                            Linear(out_channels, out_channels))
        return GINConv(mlp, **kwargs)
        
    
    def step_pool(self, pool, x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size):
        x, edge_index, batch = pool(x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size)
        return x, edge_index, batch
    
    @property
    def if_pool(self):
        return False
    
    def GNN(self, data):
        if data.batch is None:
            batch = data.edge_index.new_zeros(data.x.size(0))
        else:
            batch = data.batch
        batch_size = batch[-1] + 1
        x = data.x
        edge_index = data.edge_index
        embds = []
        num_node = scatter_add(batch.new_ones(batch.size()), batch)
        edge_batch = generate_edge_batch(edge_index, batch)
        num_edge = scatter_add(edge_batch.new_ones(edge_batch.size()), edge_batch)
        
        
        U, edge_index_list, num_nodes_tree, num_edges_tree = \
        Uext_batch(x, edge_index, batch, batch_size, num_node, num_edge, self.num_layers+1)
        if self.lin_before_conv:
            x = self.lin_start(x)
            x = F.relu(self.bn_start(x))
            
        for i in range(self.num_layers):
            
            x = self.convs[i](x=x, edge_index=edge_index)
            x = torch.relu(self.bns[i](x))
            x, edge_index, batch = self.step_pool(self.pools[i], x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size)
            embd = global_add_pool(x, batch)
            embd = F.relu(self.bns_embd[i](embd))
            if self.dropout is not None:
                embd = F.dropout(embd, p=self.dropout, training=self.training)
            embds.append(embd)   
        embds = torch.cat(embds, dim=-1)
        return embds
    
    def forward(self, data):
        self.perm = []
        self.score = []
        self.x = []
        embds = self.GNN(data)
        out = self.lin(embds)
        return out














































