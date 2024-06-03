import numpy as np
import networkx as nx
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
import torch as th
from torch import nn,int64

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing,SimpleConv,global_mean_pool,GCN
from torch_geometric.nn.pool import ASAPooling,avg_pool
from torch_geometric.utils import to_dense_adj, coalesce, remove_self_loops
from torch_geometric.seed import seed_everything
from torch_scatter import scatter_sum

from typing import List,Dict

from SEGAD_utils import SG_Encoder

# todo: 1. query时候去掉单节点图query得到的值
#       2. 输入格式，即只输入edge_index和edge_weight，编码时再输入x
#       3. x写定导致forward时候没法更新树的子图encoder，要把建树和编码分开


class EncodingLevel(nn.Module):
    def __init__(self, graph_data:Data, 
                 adj_process_method = None, 
                 optimize_p = 0.15, 
                 encoder = None,
                 if_eq_height = False,
                 if_pre_coding = True,
                 this_se = 1,
                 **kwargs
                 ):
        r"""
        Args:
            graph_data(thg.data): Each 
            adj_process: preprocess adjacency matrix
            p: parameter of SE2d clustering.
            coding_method: the method to generate codeword,
                like add node embedding via module SE weighting (add/seadd/), or subgraph proprogation(prop)
        
        """
        super(EncodingLevel, self).__init__()
        self.graph = graph_data
        if graph_data.n_id is None:
            graph_data.n_id = th.arange(graph_data.num_nodes)
        # if graph_data.pos is None:
        #     graph_data.pos = th.Tensor([range(0,graph_data.num_nodes)]).T
        self.adj_process_method = adj_process_method
        self.adj_mx = self.adj_processing()
        self.p = optimize_p
        self.if_eq_height = if_eq_height
        self.if_pre_coding = if_pre_coding

        self.encoder = encoder

        self.comms_cnt = None
        self.comms_idx:th.Tensor = None # 1dim N
        # self.comms_nodes:Dict[int,th.Tensor] = None # {N_com : n x pos_d}    # N_com个社区，每个社区n个节点，有pos_d维度的位置编码
        self.comms_nodes:Dict[int,th.Tensor] = None # {N_com : n}             # N_com个社区，每个社区n个节点 n_id
        self.sub_emb:th.Tensor = None # [N_com x code_d]                     # N_com个社区的编码
        self.comms_subgraph:List[Data] = None                           # N_com个社区，每个社区代表的节点子图
        self.coarsen_edge_index:th.Tensor = None                               # 本层划分出的N_com个社区构成的粗化子图
        self.coarsen_edge_weight:th.Tensor = None
        self.next_level:List[EncodingLevel] = None                           # N_com个社区，每个社区再次划分的层
        self.SE2d = None                                                     # 本划分的二维结构熵
        self.vertex_se = None                                                # 每个图节点的结构熵
        self.module_se:th.Tensor = None                                      # 每个社区的结构熵
        

        self.this_se = this_se
        self.if_leaf = False

        # statics
        self.coa_self_loop = True

        # todo: exclude leaf coding and subgraph sampling.
        self.process2d()

    

    def adj_processing(self):
        # todo:
        if self.adj_process_method == None:
            g = self.graph
            return to_dense_adj(edge_index=g.edge_index,
                                edge_attr=g.edge_weight if g.edge_weight is not None else g.edge_attr[0])
    
    def get_coarsen_edges(self, cluster, edge_index, edge_attr):
        num_nodes = cluster.size(0)
        edge_index = cluster[edge_index.view(-1)].view(2, -1)
        if not self.coa_self_loop:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        if edge_index.numel() > 0:
            edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        return edge_index, edge_attr
    
    def clustering(self):

        if self.graph.num_nodes == 1:
            self.SE2d = 1.
            self.vertex_se = th.tensor([1.,])
            self.module_se = th.tensor([1.,])
            self.comms_idx = th.tensor([0,],dtype=int64)
            self.comms_cnt = 1
            self.comms_nodes = comms = {0:self.graph.n_id[:]}
            self.if_leaf =True
            return comms
        
        p = self.p
        edges = self.graph.edge_index.t()
        weights = self.graph.edge_weight
        dist = scatter_sum(weights, edges[:, 1]) + scatter_sum(weights, edges[:, 0])  # dist/2=di
        dist = dist / (2 * weights.sum())  # ew.sum()=vol(G) dist=di/vol(G)

        gs = GraphSparse(edges, weights, dist)
        optim = OperatorPropagation(Partitioning(gs, None))
        optim.perform(p)
        division = optim.enc.node_id
        comms_ids = th.unique(division)
        comms_cnt = comms_ids.shape[0]
        
        
        self.SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
        self.module_se = module_se = optim.enc.structural_entropy(reduction='module', norm=True)
        #### debug:
        self.vertex_se = optim.enc.structural_entropy(reduction='vertex', norm=True) 


        comms = {}

       
        for i in comms_ids:
            idx = division == i
            if idx.any():
                # comms[i] = idx.nonzero().squeeze(1)
                comms[int(i)] = self.graph.n_id[idx.nonzero().squeeze(1)]


        self.comms_idx = division
        self.comms_cnt = comms_cnt
        self.comms_nodes = comms
        self.coarsen_edge_index,self.coarsen_edge_weight = self.get_coarsen_edges(
            division,self.graph.edge_index,self.graph.edge_weight)

        return comms
    
    def subgraph_sampling(self):
        g = self.graph
        assert self.comms_nodes is not None,'need to cluster'
        
        subgraphs = []
        comms_cnt = self.comms_cnt
        comms_idx = self.comms_idx
        for comms_id in range(comms_cnt):
            sub_ids =  comms_idx == comms_id
            if sub_ids.any():
                subgraphs.append(g.subgraph(sub_ids.nonzero().squeeze(1)))

        self.comms_subgraph = subgraphs
        return subgraphs
    
    def coding(self,x=None,id_mappping=None):
        xt = x if x is not None else self.graph.x

        if self.encoder is None:
            embed = global_mean_pool(xt,batch=self.comms_idx)
        
        else:
            embed = th.tensor([])
            for com in range(self.comms_cnt):
                idx = self.comms_idx == com
                if idx.any():
                # comms[i] = idx.nonzero().squeeze(1)
                    x0 = xt[idx.nonzero().squeeze(1)]
                    # htt = self.encoder(self.comms_subgraph[com].x,
                    #                    self.comms_subgraph[com].edge_index,
                    #                    self.comms_subgraph[com].edge_weight)
                    h = self.encoder(x0,
                                self.comms_subgraph[com].edge_index,
                                self.comms_subgraph[com].edge_weight)
                    
                embed = th.cat((embed,h.unsqueeze(0)),dim=0)
        
        self.sub_emb = embed
        return embed

    def tree_coding(self,x,id_mapping):
        # todo: calculate new embedding for whole tree using self.encoder.
        self.coding(x)
        if self.next_level is not None:
            for com in range(self.comms_cnt):
                com_n_ids = self.comms_nodes[com]
                indices = th.where(th.eq(com_n_ids[:, None], id_mapping))[1]
                x_next = x[indices]
                self.next_level[com].tree_coding(x_next,com_n_ids)

    def process2d(self):
            self.clustering()
            self.subgraph_sampling()
            if self.if_pre_coding:
                self.coding()

    def expanding_tree(self,**kwargs):
        if self.next_level is not None:
            for each in self.next_level:
                each.expanding_tree(**kwargs)
        else:
            if self.if_leaf is True and self.if_eq_height is False:
                return
            else:
                temp_level = []
                for sub in range(self.comms_cnt):
                    codebook= EncodingLevel(graph_data=self.comms_subgraph[sub],
                        adj_process_method = self.adj_process_method, 
                        optimize_p = self.p, 
                        encoder = self.encoder,
                        if_eq_height = self.if_eq_height,
                        if_pre_coding=self.if_pre_coding,
                        this_se = self.module_se[sub],
                        **kwargs
                    )
                    codebook.process2d()
                    temp_level.append(codebook)
                self.next_level = temp_level
            
    def query(self,n_ids,mode=None,**kwargs):          
        # exclude leaf

        if self.if_leaf == True:
            return ([th.Tensor(),],[]) if mode=='full' else [th.Tensor(),]
            

        if self.next_level is None:
            code_list = [None for _ in range(len(n_ids))] 
            if mode == 'full':
                modules_list = [None for _ in range(len(n_ids))] 

            for com in range(self.comms_cnt):
                indices = th.where(th.eq(self.comms_nodes[com][:, None], n_ids))[1]

                
                if indices.shape[0] > 0:
                    if mode == 'full':
                        for i in indices:
                            code_list[i] = self.sub_emb[com].unsqueeze(0)
                            modules_list[i] = [self,]
                    else:
                        for i in indices:
                            code_list[i] = self.sub_emb[com].unsqueeze(0)
            
            return (code_list,modules_list) if mode=='full' else code_list
        

        else:
            code_list = [None for _ in range(len(n_ids))] 
            if mode == 'full':
                modules_list = [None for _ in range(len(n_ids))] 

            
            for com in range(self.comms_cnt):
                indices = th.where(th.eq(self.comms_nodes[com][:, None], n_ids))[1]
                new_n_ids = n_ids[indices]
                
                if indices.shape[0] > 0:
                    if mode == 'full':
                        next_code_list,next_modules_list = self.next_level[com].query(new_n_ids,mode)

                        for i in indices:
                            code_list[i] = self.sub_emb[com].unsqueeze(0)
                            modules_list[i] = [self,]

                        # code_list[indices] += next_code_list
                        modules_list[indices] += next_modules_list

                        jj=0
                        for ii in indices:
                            modules_list[ii] += next_modules_list[jj]
                            code_list[ii] = th.cat((code_list[ii],next_code_list[jj]),dim=0)
                            jj += 1
                        

                    else:
                        next_code_list = self.next_level[com].query(new_n_ids,mode)

                        for i in indices:
                            code_list[i] = self.sub_emb[com].unsqueeze(0)

                        jj=0
                        for ii in indices:
                            code_list[ii] = th.cat((code_list[ii],next_code_list[jj]),dim=0)
                            jj += 1
            
            return (code_list,modules_list) if mode=='full' else code_list

    def forward(self,x):
        # encoding feature x  

        
        return




def my_test():
    subgraphenc = SG_Encoder(
        in_dim = 4,
        hid_dim = 16,
        out_dim = 4,
        num_layers = 1,
        act = nn.functional.relu,
        g_norm = None,
        n_norm = nn.LayerNorm,
        backbone = GCN,
        dropout = 0.01,
        agg = 'att',
    )

    tempcb = EncodingLevel(
                 graph_data=temp_graph, 
                 adj_process_method = None, 
                 optimize_p = 0.15, 
                 encoder = subgraphenc,
                 if_eq_height = False,
                 if_pre_coding= False,
                 this_se = 1)
    tempcb.expanding_tree()
    tempcb.expanding_tree()
    tempcb.expanding_tree()
    tempcb.tree_coding(x=temp_graph.x,id_mapping=temp_graph.n_id)
    fincode = tempcb.query(th.tensor([15,12, 8 ,21],dtype=th.int64))
    for each in fincode:
        print(each)
        print(each.shape)


    

if __name__ == '__main__':
    from torch_geometric.datasets import FakeDataset,RandomPartitionGraphDataset
    seed_everything(42)
    temp_dataset = FakeDataset(num_graphs=1,avg_num_nodes=60,avg_degree=3.5,num_channels=4,
                             edge_dim=1,num_classes=4,is_undirected=True)
    temp_graph = temp_dataset.generate_data()
    temp_graph.n_id = th.tensor(range(0,temp_graph.num_nodes))

    my_test()

    

        





