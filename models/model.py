import os
import dgl
import math
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
from collections import defaultdict
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
from utils.graph_util import build_adj_from_etype, normalize_graph_mat
import matplotlib.pyplot as plt

class Multi_interest_Loss(nn.Module):
    def __init__(self,loss='em_posterior', reduction='sum'): 
        super().__init__()
        self.loss = loss
        self.reduction = reduction
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, graph, etype):
        pred_score_e = graph.edges[etype].data['pred_score_e']
        em_posterior = graph.edges[etype].data[self.loss] # percentile em_posterior
        per_edge_loss_e = F.binary_cross_entropy_with_logits(
            pred_score_e,
            em_posterior,
            reduction='none'
        )
        return per_edge_loss_e

class Multi_interest_fit(nn.Module):
    def __init__(self, dim, loss='em_posterior'): # em_posterior percentile
        super().__init__()
        self.loss_fn = Multi_interest_Loss(loss=loss)
        
    def edge_score(self, edges):
        h_u = edges.src['h']   # (E, D)
        h_v = edges.dst['h']   # (E, D)
        mf_score = (h_u * h_v).sum(dim=-1)
        score_e = torch.sigmoid(mf_score)
        return {'pred_score_e': score_e}

    def forward(self, graph, h_u, h_i, etype):
        with graph.local_scope():
            graph.nodes['user'].data['h'] = h_u
            graph.nodes['game'].data['h'] = h_i
            graph.apply_edges(self.edge_score, etype = etype)
            loss = self.loss_fn(graph, etype)
        return loss
            
class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_20, device, gamma=80, ablation=False):
        super().__init__()
        print("\n=== Graph Information ===")
        print("Node types:", graph.ntypes)  
        print("Edge types:", graph.etypes)  
        print("Canonical edge types:", graph.canonical_etypes)  
        print("\n=== Node Statistics ===")
        for ntype in graph.ntypes:
            print(f"Number of {ntype} nodes:", graph.number_of_nodes(ntype))
        print("\n=== Edge Statistics ===")
        for etype in graph.etypes:
            print(f"Number of {etype} edges:", graph.number_of_edges(etype))
        self.ablation = ablation
        self.device_ = torch.device(device)
        torch.cuda.empty_cache()
        self.args = args
        self.hid_dim = args.embed_size  
        self.layer_num_user_game = args.layers_user_game
        self.graph = graph.to(self.device_)
        self.graph_20 = graph_20.to(self.device_)
        
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)
        self.graph_item2type = dgl.edge_type_subgraph(self.graph_20,['genre']).to(self.device_)
        self.graph_type2item = dgl.edge_type_subgraph(self.graph_20,['genred']).to(self.device_)
        triplets = []
        if args.w_CI:
            save_path = '../data_exist/triplets.pkl'
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    triplets = pickle.load(f)
                print(f"直接加载 triplets...负采样共{len(triplets)}对")
            else:
                print("重新计算 triplets...")
                u, v = self.graph_user2item.edges(etype='play')
                t = self.graph_user2item.edges['play'].data['percentile']
                e = self.graph_user2item.edges['play'].data['em_posterior']
                
                user_dict = defaultdict(list)
                for uid, iid, time, em_posterior in zip(u, v, t, e):
                    user_dict[int(uid)].append((int(iid), float(time), float(em_posterior)))
                
                for u in user_dict:
                    items = user_dict[u]
                    items.sort(key=lambda x: (x[2], x[1]))
                    # items = sorted(items, key=itemgetter(2,1))
                    for i in range(len(items)-1):
                        low_item, low_time, low_em = items[i]
                        high_item, high_time, high_em = items[i+1]

                        if low_time == high_time and low_em == high_em:
                            continue

                        triplets.append([u, high_item, low_item, high_em, low_em])
                # 保存
                with open(save_path, 'wb') as f:
                    pickle.dump(triplets, f)
                
        self.u_idx = torch.tensor([x[0] for x in triplets])
        self.pos_idx = torch.tensor([x[1] for x in triplets])
        self.neg_idx = torch.tensor([x[2] for x in triplets])
        S_graph_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../data_exist/old_genre/S_graph.bin")
        S_graph, _ = dgl.load_graphs(S_graph_path)
        self.graph_S = S_graph[0].to(self.device_)
        self.graph_S = dgl.edge_type_subgraph(self.graph_S, [('user','play','game'),('game','played by','user')])
        print(f"Loading graph_s from: {S_graph_path}")

        seek_graph_path = "../data_exist/mrw.bin"
        seek_plays_weight_path = "../data_exist/weights_mrw.pth"
        seek_graph, _ = dgl.load_graphs(seek_graph_path)
        self.graph_seek=seek_graph[0].to(self.device_)
        self.graph_seek = dgl.edge_type_subgraph(self.graph_seek, [('user','plays','game'),('game','played_by','user')])
        self.weight_edge = torch.load(seek_plays_weight_path).to(self.device_)
        print(f"Loading seek graph from: {seek_graph_path}")
        print(f"Loading seek weights from: {seek_plays_weight_path}")
        self.graph_item2user_S = dgl.edge_type_subgraph(self.graph_S,['played by']).to(self.device_)
        self.graph_user2item_S = dgl.edge_type_subgraph(self.graph_S,['play']).to(self.device_)

        self.graph_item2user_seek = dgl.edge_type_subgraph(self.graph_seek,['played_by']).to(self.device_)
        self.graph_user2item_seek = dgl.edge_type_subgraph(self.graph_seek,['plays']).to(self.device_)

        # 模型准备
        self.edge_node_weight =True
        # id embedding
        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)
        self.preference_embedding = torch.nn.Parameter(torch.randn(20, self.hid_dim)).to(torch.float32)
        self.conv_type = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree=True).to(self.device_)
        
        self.ln_user_f = nn.LayerNorm(self.hid_dim)
        self.ln_user_s = nn.LayerNorm(self.hid_dim)
        
        self.build_model_ssl()
        self.build_model_S()
        self.multi_interest_fit_f = Multi_interest_fit(dim=self.hid_dim)
        self.multi_interest_fit_s = Multi_interest_fit(dim=self.hid_dim)
        
    def _build_layers(self):
        layers = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            if self.edge_node_weight:
                conv = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree=True)
            else:
                conv = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree=True),
                    'played by': GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree=True)
                })
            layers.append(conv)
        return layers.to(self.device_)
    
    def build_model_item(self, graph_item):
        self.sub_g1 = dgl.edge_type_subgraph(graph_item,['co_genre']).to(self.device_)

    def build_model_ssl(self):
        self.layers = self._build_layers()

    def build_model_S(self):
        self.layers_S = self._build_layers()
    
    def forward(self,epoch):
                
        h_f = {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        h_s= {'user':self.user_embedding.clone(), 'game':self.item_embedding.clone()}
        loss_mutil_inter_f = 0
        loss_mutil_inter_s = 0
        loss_mutil_inter_f_list = []
        loss_mutil_inter_s_list = []
        
        L = len(self.layers)
        self.lambda0 = 0.01   # 超参数
        self.p = 1.1  
        
        for i,layer in enumerate(self.layers):
            l = i + 1
            lambda_l = self.lambda0 * ((l / L) ** self.p)
            
            if self.edge_node_weight == True:
                
                h_user_f = layer(self.graph_item2user, (h_f['game'],h_f['user']))
                h_item_f = layer(self.graph_user2item, (h_f['user'],h_f['game']))
                h_user_seek = layer(self.graph_item2user_seek, (h_f['game'],h_f['user']),edge_weight=self.weight_edge)
                h_item_seek = layer(self.graph_user2item_seek, (h_f['user'],h_f['game']))
                    
                if self.args.w_CI:
                    
                    u_emb = h_user_f[self.u_idx]
                    i_emb = h_item_f[self.pos_idx]
                    j_emb = h_item_f[self.neg_idx]
                    
                    pos_score = (u_emb * i_emb).sum(dim=-1)
                    neg_score = (u_emb * j_emb).sum(dim=-1)
                    loss_f_rel = torch.nn.functional.softplus(-(pos_score - neg_score)).sum()

                    loss_f_abs = self.multi_interest_fit_f(
                        self.graph_user2item,
                        h_user_f,
                        h_item_f,
                        etype='play'
                    )
                    loss_mutil_inter_f_list.append(lambda_l * (loss_f_abs + loss_f_rel))
                        
                h_user_s = layer(self.graph_item2user_S, (h_s['game'],h_s['user'])) # em_posterior   percentile
                h_item_s = layer(self.graph_user2item_S, (h_s['user'],h_s['game']))

                if self.args.w_CI:
                    h_user_s = F.normalize(h_user_s, dim=1)
                    loss_s = self.multi_interest_fit_s(   
                        self.graph_user2item_S,
                        h_user_s,
                        h_item_s,
                        etype='play'
                    )
                loss_mutil_inter_s_list.append(lambda_l * loss_s)
                
                self.args.alpha = 1.0
                h_user_f = h_user_f * self.args.alpha + h_user_seek
                h_item_f = h_item_f * self.args.alpha + h_item_seek
                h_user_s = h_user_s * self.args.alpha + h_user_seek
                h_item_s = h_item_s * self.args.alpha + h_item_seek
                h_f['user'] = h_user_f
                h_f['game'] = h_item_f 
                h_s['user'] = h_user_s 
                h_s['game'] = h_item_s 
        
        if len(loss_mutil_inter_f_list) > 0:
            loss_mutil_inter_f = torch.stack(loss_mutil_inter_f_list).mean()
        if len(loss_mutil_inter_s_list) > 0:
            loss_mutil_inter_s = torch.stack(loss_mutil_inter_s_list).sum()
        
        h_type = self.preference_embedding
        h_user_s_norm = F.normalize(h_s['user'], dim=1)
        h_type_norm_s = F.normalize(h_type, dim=1)

        tau = 2.0
        alpha = 1
        
        sim = (h_user_s_norm @ h_type_norm_s.T) / (tau * math.sqrt(self.hid_dim))
        h_t = F.softmax(sim, dim=1)
        h_t = F.dropout(h_t, p=0.3, training=self.training)

        h_user_f = h_user_f + alpha * (h_t @ h_type_norm_s)
        h_user_f = F.normalize(h_user_f, dim=1)
        h_f['user'] = h_user_f            
        return h_f,h_f,h_s,loss_mutil_inter_f,loss_mutil_inter_s


class SSLoss():
    def __init__(self,args):
        super(SSLoss, self).__init__()
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = 1
        self.ssl_game_weight=args.ssl_game_weight

    def forward(self,ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
                ia_embeddings_sub2):
        eps = 1e-8
        user_emb1 = ua_embeddings_sub1
        user_emb2 = ua_embeddings_sub2  
        normalize_user_emb1 = F.normalize(user_emb1, dim=1)
        normalize_user_emb2 = F.normalize(user_emb2, dim=1)
        normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)
        pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(normalize_user_emb1,normalize_all_user_emb2.T)
        ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  
        ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
        
        ssl_loss_item = 0
        if ia_embeddings_sub1.size(0) != 0:
            item_emb1 = ia_embeddings_sub1
            item_emb2 = ia_embeddings_sub2
            normalize_item_emb1 = F.normalize(item_emb1, dim=1)
            normalize_item_emb2 = F.normalize(item_emb2, dim=1)
            normalize_all_item_emb2 = F.normalize(ia_embeddings_sub2, dim=1)
            pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
            ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)
            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)
            ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item)) * self.ssl_game_weight

        loss=ssl_loss_item+ssl_loss_user

        return loss
    
