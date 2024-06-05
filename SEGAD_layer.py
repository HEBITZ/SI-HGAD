import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch_geometric.data import Data
from torch_geometric.nn import GCN, SAGEConv, PNAConv
from torch_geometric.utils import to_undirected, add_self_loops
from Hier_Encoding import EncodingLevel
from SEGAD_utils import FNN_Encoder,SG_Encoder,KL_SG_loss


def Hier_Construction_and_Sampling(temp_graph,
                                   num_enc_tree_level,
                                   q_node:torch.Tensor,
                                   sample_edge_num=5,
                                   subgraphenc =  SG_Encoder(
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
                                            )):

    temp_graph.n_id = torch.tensor(range(0,temp_graph.num_nodes))

    tempcb = EncodingLevel(
                 graph_data=temp_graph, 
                 adj_process_method = None, 
                 optimize_p = 0.15, 
                 encoder = subgraphenc,
                 if_eq_height = False,
                 if_pre_coding= False,
                 this_se = 1)
    for _ in range(num_enc_tree_level-1):
        tempcb.expanding_tree()

    tempcb.tree_coding(x=temp_graph.x,id_mapping=temp_graph.n_id)
    samp_code,_,subgraphs = tempcb.query(q_node,mode='full')

    samp_mean = []
    samp_sigma = []
    samp_embed = []
    for i_subs in subgraphs:
        samp_i_mean = None
        samp_i_sigma = None
        samp_i_embed = None
        for subgraph in i_subs:
            g_mean,g_sigma,g_embed = edge_sampling(subgraph,sample_edge_num)
            samp_i_mean = g_mean if samp_i_mean is None else torch.cat((samp_i_mean,g_mean),dim=0)
            samp_i_sigma = g_sigma if samp_i_sigma is None else torch.cat((samp_i_sigma,g_sigma),dim=0)
            samp_i_embed = g_embed if samp_i_embed is None else torch.cat((samp_i_embed,g_embed),dim=0)
        samp_embed.append(samp_i_embed)
        samp_mean.append(samp_i_mean)
        samp_sigma.append(samp_i_sigma)


    return samp_code,samp_mean,samp_sigma,samp_embed


def edge_sampling(subgraph:Data,sample_edge_num):
    edge_index = subgraph.edge_index
    edge_attr = subgraph.edge_attr if subgraph.edge_attr is not None else torch.empty([sample_edge_num,0])
    edge_weight =subgraph.edge_weight if subgraph.edge_weight is not None else torch.empty([sample_edge_num,0])
    node_feat = subgraph.x

    rand_idx = torch.randint(0, edge_index.shape[1], (sample_edge_num,))
    sampled_edge_index = ((edge_index.T)[rand_idx]).T
    sampled_edge_attr = edge_attr[rand_idx]
    sampled_edge_weight = edge_weight[rand_idx]
    sampled_edge_embed = node_feat[sampled_edge_index]
    sampled_edge_embed = torch.cat([sampled_edge_embed[0].squeeze(),
                                    sampled_edge_embed.squeeze(),
                                    sampled_edge_attr,
                                    sampled_edge_weight.unsqueeze(dim=-1)],dim=-1)
    # embed_dim = sampled_edge_embed.shape[-1]
    
    sampled_edge_mean = torch.mean(sampled_edge_embed,dim=0)
    sampled_edge_sigma = torch.std(sampled_edge_embed,dim=0)
    return sampled_edge_mean,sampled_edge_sigma,sampled_edge_embed
    


class Hier_Recon(nn.Module):
    def __init__(
            self,
            temp_graph,
            encoding_tree,
            recon_len,
            hid_dim,
            # output_node_dim,
            output_edge_dim,
            sample_size,

            device = torch.device('cpu'),
            dropout_rate = 0.01,
            act_func=torch.nn.functional.relu,

            **kwargs):
        super(Hier_Recon, self).__init__()
        self.device = device
        self.temp_graph = temp_graph,
        self.encoding_tree = encoding_tree,
        self.sample_size = sample_size
        self.recon_len = recon_len
        self.output_edge_dim =output_edge_dim

        self.m_minibatch = torch.distributions.Normal(
            torch.zeros(sample_size,hid_dim),torch.ones(sample_size,hid_dim))
        self.sg_dist_mean_decoder = nn.Linear(hid_dim, hid_dim)
        self.sg_dist_sigma_decoder = nn.Linear(hid_dim, hid_dim)
        self.sg_dist_decoder = nn.Linear(hid_dim, output_edge_dim)
        self.se_decoder = FNN_Encoder(hid_dim, hid_dim, 1, dropout=dropout_rate)
        self.sg_feat_decoder = FNN_Encoder(hid_dim, hid_dim,
                                          hid_dim, dropout=dropout_rate)
        # self.sg_feat_expand = nn.Linear(hid_dim,output_node_dim)
        # self.sg_dist_expand = nn.Linear(hid_dim,output_edge_dim)


        self.se_loss_func = nn.MSELoss()
        self.sg_feat_loss_func = nn.MSELoss()
        self.struc_dist_loss = KL_SG_loss()
        

    
    def forward(self,
                hid_x,
                hid_x_idx,
                ):
        
        # hid_x = hid_x[hid_x_idx]

        se_recon_list = []
        sg_feat_recon_list = []
        sg_dist_recon_list = []
        
        for i in hid_x_idx:
            hid_i = xi_code = hid_x[i]
            i_se_recon = []
            i_sg_feat_recon = []
            i_sg_dist_recon = []
            
            for j in range(len(self.recon_len[i]),0):
                
                recon_se = self.se_decoder(xi_code)
                i_se_recon.insert(0,recon_se)

                hat_mean = xi_code.repeat(self.sample_size, 1)
                hat_mean = self.sg_dist_mean_decoder(hat_mean)
                hat_sigma = xi_code.repeat(self.sample_size, 1)
                hat_sigma = self.sg_dist_sigma_decoder(hat_sigma)
                std_z = self.m_minibatch.sample().to(self.device)
                var = hat_mean + hat_sigma.exp() * std_z
                recon_sg = self.sg_dist_decoder(var)
                

                sum_neighbor_norm = 0
            
                for _, recon_edge in enumerate(recon_sg):
                    sum_neighbor_norm += torch.norm(recon_edge) / math.sqrt(self.output_edge_dim)
                recon_sg = torch.unsqueeze(recon_sg, dim=0).to(self.device)
                i_sg_dist_recon.insert(0,recon_sg)


                xi_code = self.sg_feat_decoder(xi_code)
                i_sg_feat_recon.insert(0,xi_code)
            
            se_recon_list.append(i_se_recon)
            sg_feat_recon_list.append(i_sg_feat_recon)
            sg_dist_recon_list.append(i_sg_dist_recon)
        
        return sg_dist_recon_list,sg_feat_recon_list,se_recon_list


    def cal_hier_loss(self,
                      recon_idx,
                      samp_se_list,
                      samp_code_list,
                      samp_dist_list,
                      recon_se_list,
                      recon_sg_feat_list,
                      recon_sg_dist_list,
                      degree_list,
                      volG,
                      xi1=0.2,
                      xif=0.5,
                      xid=0.5):


        
        idx_loss_list = []
        for i in recon_idx:
            tot_loss = 0.0
            for level in range(self.recon_len[i],0):
                recon_se_loss = self.se_loss_func(samp_se_list[i][level],
                                                  recon_se_list[i][level])
                recon_feat_loss = self.sg_feat_loss_func(samp_code_list[i][level],
                                                         recon_sg_feat_list[i][level])
                recon_dist_loss = self.struc_dist_loss(samp_dist_list[i][level],
                                                       recon_sg_dist_list[i][level])
                level_loss = xi1*recon_se_loss + xif*recon_feat_loss + xid * recon_dist_loss
                tot_loss += (degree_list[i][level]/volG) * level_loss
            idx_loss_list.append(tot_loss)
        batch_loss = 0
        for each in idx_loss_list:
            batch_loss += each
        return idx_loss_list, batch_loss
                


        




        ground_truth_degree_matrix = \
            torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits,
                                            ground_truth_degree_matrix.float())
        degree_loss_per_node = \
            (degree_logits-ground_truth_degree_matrix).pow(2)
        
        h_loss = 0
        feature_loss = 0
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for t in range(self.sample_time):
            # feature reconstruction loss 
            h0_prime = feat_recon_list[t]
            feature_losses_per_node = (h0-h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)
            
            # neigbor distribution reconstruction loss
            if self.full_batch:
                # full batch neighbor reconstruction
                det_target_cov, det_generated_cov, h_dim, trace_mat, z = \
                                                        neigh_recon_list[t]
                KL_loss = 0.5 * (torch.log(det_target_cov / 
                                           det_generated_cov) - \
                        h_dim + trace_mat.diagonal(offset=0, dim1=-1, 
                                                    dim2=-2).sum(-1) + z)
                local_index_loss = torch.mean(KL_loss)
                local_index_loss_per_node = KL_loss
            else: # mini batch neighbor reconstruction
                local_index_loss = 0
                local_index_loss_per_node = []
                gen_neighs, tar_neighs, mask_lens = neigh_recon_list[t] 
                for generated_neighbors, target_neighbors, mask_len in \
                                        zip(gen_neighs, tar_neighs, mask_lens):
                    temp_loss = self.neighbor_loss(generated_neighbors,
                                                   target_neighbors,
                                                   mask_len,
                                                   self.device)
                    local_index_loss += temp_loss
                    local_index_loss_per_node.append(temp_loss)
                                        
                local_index_loss_per_node = \
                    torch.stack(local_index_loss_per_node)
            
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)
            
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        
        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)
        
        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),
                                           dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
                
        h_loss_per_node = h_loss_per_node.reshape(batch_size, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(batch_size, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(batch_size, 1)
        
        loss = self.lambda_loss1 * h_loss \
            + degree_loss * self.lambda_loss3 \
            + self.lambda_loss2 * feature_loss
        
        loss_per_node = self.lambda_loss1 * h_loss_per_node \
            + degree_loss_per_node * self.lambda_loss3 \
                + self.lambda_loss2 * feature_loss_per_node
        
        return loss, loss_per_node, h_loss_per_node, \
            degree_loss_per_node, feature_loss_per_node
        

        return fin_hier_loss



    
