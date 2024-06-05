import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
import math


def KL_SG_loss(predictions, targets, mask_len, device):

    x1 = predictions.squeeze().cpu().detach()[:mask_len, :]
    x2 = targets.squeeze().cpu().detach()[:mask_len, :]

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
    cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim
    + torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) + (mean_x2 - 
    mean_x1).reshape(1,-1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - 
                                                                mean_x1))
    KL_loss = KL_loss.to(device)
    return KL_loss

class FNN_Encoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim = 64,
                 out_dim = 64,
                 num_layers = 1,
                 act=nn.functional.relu,
                 norm=nn.BatchNorm1d,
                 dropout = 0.0,
                 **kwargs):
        super(FNN_Encoder, self).__init__()

        self.act = act
        self.linears = nn.ModuleList()
        self.norm = norm
        self.batch_norms = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)

        assert num_layers >= 1, 'Layer need to be positive'

        if num_layers == 1:
            self.linears.append(nn.Linear(in_dim,out_dim))
        else:
            self.linears.append(nn.Linear(in_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.linears.append(nn.Linear(hid_dim, out_dim))

        for _ in range(num_layers - 1):
            self.batch_norms.append(self.norm((hid_dim)))
        
    def forward(self,x):
        h = x
        for layer in range(self.num_layers - 1):
            h = self.act(self.batch_norms[layer](self.linears[layer](h)))
        return self.dropout_layer(self.act(self.linears[-1](h)))
        

class SG_Encoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers = 1,
                 act = nn.functional.relu,
                 g_norm = None,
                 n_norm = nn.LayerNorm,
                 backbone = GCN,
                 dropout = 0.0,
                 jk = 'lstm',
                 agg = 'sum',
                 **kwargs):
        super(SG_Encoder, self).__init__()
        self.hid_dim =hid_dim
        self.agg = agg
        self.act = act
        self.norm_layer = n_norm(hid_dim)
        self.dropout_layer = nn.Dropout(dropout)


        self.GNN_encoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=num_layers,
            out_channels=hid_dim,
            dropout=dropout,
            norm=g_norm,
            act=act,
            jk=jk,
            **kwargs
        )

        self.fc_out = nn.Linear(hid_dim,hid_dim)
        self.linear_out = nn.Linear(hid_dim,out_dim)

        if agg == 'sum':
            self.linear_in = nn.Linear(in_dim,hid_dim)
            
        elif agg == 'att':
            self.num_heads = kwargs.pop('num_heads',4)
            self.scale = kwargs.pop('scale',None)
            self.head_dim = self.hid_dim // self.num_heads

            assert (
                self.head_dim * self.num_heads == self.hid_dim
            ), "hid dim needs to be divisible by num_heads"

            self.v_proj = nn.Linear(in_dim,hid_dim)
            self.q_proj = nn.Linear(in_dim,hid_dim)  
            self.k_proj = nn.Linear(in_dim,hid_dim)
            
    
    def forward(self,x:th.Tensor,edge_index,edge_weight):
        
        if self.agg == 'sum':
            h = skip0 = self.linear_in(x)
            h = self.GNN_encoder(h,edge_index,edge_weight)
            h = self.fc_out(h)
            h = self.dropout_layer(self.norm_layer(h+skip0))
            
        
        elif self.agg == 'att':
            num_nodes = x.shape[0]

            value:th.Tensor = self.v_proj(x)
            key:th.Tensor = self.k_proj(x)
            query = skip0 = self.q_proj(x)

            query = skip1 = self.GNN_encoder(query,edge_index,edge_weight) + query


            value = value.reshape(num_nodes, self.num_heads, self.head_dim)
            key = key.reshape(num_nodes, self.num_heads, self.head_dim)
            query = query.reshape(num_nodes, self.num_heads, self.head_dim)

            energy = th.einsum("qhd,khd->qkh", [query, key])

            scale = self.scale if self.scale is not None else 1/(self.hid_dim ** (1 / 2))
            # scale = num_nodes*th.log(avg_node)/(self.hid_dim ** (1 / 2))

            attention = th.softmax(energy * scale, dim=1)

            h = th.einsum("qkh,khd->qhd", [attention, value]).reshape(
                num_nodes, self.num_heads * self.head_dim
            )

            h = self.fc_out(h)

            h = self.dropout_layer(self.norm_layer(h+skip0+skip1))
            
        h = th.sum(h,dim=0)
        out = self.linear_out(h)

        return out


