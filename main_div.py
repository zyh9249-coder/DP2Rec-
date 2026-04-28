import sys
sys.path.append('../')
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import logging

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
from utils.parser_div import parse_args
from dgl.nn.pytorch.conv import GraphConv
from utils.dataloader_steam import Dataloader_steam_filtered

from utils.dataloader_item import Dataloader_item_graph
from models.model import Proposed_model
from models.model import SSLoss
from models.Predictor import Predictor
import pickle
import torch.nn.functional as F
import time
# 主日志目录
base_log_dir = "输入你的log文件夹"
os.makedirs(base_log_dir, exist_ok=True)

existing_experiments = [d for d in os.listdir(base_log_dir) if d.startswith("experiment_")]
existing_indices = [int(d.split("_")[1]) for d in existing_experiments if d.split("_")[1].isdigit()]
next_index = max(existing_indices, default=-1) + 1
experiment_dir = os.path.join(base_log_dir, f"experiment_{next_index}")
os.makedirs(experiment_dir, exist_ok=True)

# 日志文件路径
log_file = os.path.join(experiment_dir, "log.txt")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置全局日志级别

# 文件日志 handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# 控制台日志 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# 添加 handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 测试日志
logger.info(f"Logging to {log_file} and console.")

ls_5 = []
ls_10 = []
ls_20 = []

def orthogonal_loss(L):
    # L: [K, D]
    L_norm = F.normalize(L, dim=1)
    gram = torch.matmul(L_norm, L_norm.T)  # [K, K]
    I = torch.eye(L.size(0), device=L.device)
    # loss = ((gram - I) ** 2).mean()
    loss = 0.5 * ((gram - I) ** 2).sum()
    return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_valid_mask(DataLoader, graph, valid_user):
    data_exist_dir = os.path.join("data_exist")
    path_valid_mask_trail = path+"/valid_mask.pth"
    if os.path.exists(path_valid_mask_trail):
        valid_mask = torch.load(path_valid_mask_trail)
        return valid_mask
    else:
        valid_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
        for i in range(len(valid_user)):
            user = valid_user[i]
            item_train = torch.tensor(DataLoader.dic_user_game[user])
            valid_mask[i, :][item_train] = 1
        valid_mask = valid_mask.bool()
        torch.save(valid_mask, path_valid_mask_trail)
        return valid_mask

def construct_negative_graph(graph, etype,device):

    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    src = src.to(device)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape).to(device)
    return dst, dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def get_coverage(ls_tensor, mapping):
    covered_items = set()
    
    for i in ls_tensor:
        if int(i) in mapping.keys():
            types = mapping[int(i)]
            covered_items = covered_items.union(set(types))
    
    return float(len(covered_items))

def get_category_entropy(ls_tensor, mapping):
    category_counts = {}
    total_count = 0
    
    # 统计各类别的出现频次
    for i in ls_tensor:
        if int(i) in mapping.keys():
            types = mapping[int(i)]
            # 如果类型是单个整数，转换为列表处理
            if isinstance(types, int):
                types = [types]
            # 如果类型是集合，转换为列表
            elif isinstance(types, set):
                types = list(types)
            
            # 统计每个类别的频次
            for category in types:
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1
                total_count += 1
    
    # 如果没有有效的类别，返回0
    if total_count == 0:
        return 0.0
    
    # 计算类别熵
    entropy = 0.0
    for count in category_counts.values():
        prob = count / total_count
        if prob > 0:  # 避免log(0)
            entropy -= prob * torch.log2(torch.tensor(prob))
    
    return float(entropy)

def validate(valid_mask, dic, h, ls_k, genre_mapping, to_get_coverage,device):
    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[valid_mask] = -float('inf')
    valid_mask = torch.zeros_like(valid_mask)
    # dic 测试集用户交互的游戏列表
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1
    _, indices = torch.sort(rating, descending = True)

    indices = indices.to(device)
    valid_mask = valid_mask.to(device)
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()
    res = []
    ndcg = 0
    for k in ls_k:
        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        ideal = ideal.to(device)
        discount = discount.to(device)
        idcg = (ideal[:, :k] / discount).sum(dim = 1)
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)
        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        
        precision = torch.mean(result[:, :k].mean(1))
        
        if to_get_coverage == False:
            coverage = -1
        else:
            cover_tensor = torch.tensor([get_coverage(indices[i,:k], genre_mapping) for i in range(users.shape[0])])
            coverage = torch.mean(cover_tensor)
            entropy_tensor = torch.tensor([get_category_entropy(indices[i,:k], genre_mapping) for i in range(users.shape[0])])
            category_entropy = torch.mean(entropy_tensor)
        
        logging_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}, coverage = {}, category_entropy = {}".format(
            k, ndcg, recall, hit, precision, coverage, category_entropy)
        
        logging.info(logging_result)
        res.append(logging_result)
    
    return coverage, str(res)

if __name__ == '__main__':

    seed=int(2025)
    setup_seed(seed)
    args = parse_args()
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    path = "./steam_data"

    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genres_path = path + '/Games_Genres.txt'

    DataLoader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, genres_path)
    graph = DataLoader.graph.to(device)
    
    DataLoader_item = Dataloader_item_graph( app_id_path, genres_path, DataLoader)
    graph_item_item = DataLoader_item.graph_item

    graph = dgl.edge_type_subgraph(graph, [('user','play','game'),('game','played by','user')])
    
    valid_user = list(DataLoader.valid_data.keys())
    valid_mask = get_valid_mask(DataLoader, graph, valid_user)

    model = Proposed_model(args, graph, graph_item_item,
                        device, gamma=args.gamma, ablation = False)
    model.to(device)    
    ssloss = SSLoss(args)
    predictor = Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    stop_count = 0
    ls_k = args.k
    total_epoch = 0  
    loss_pre = float('inf')
    loss = 0
    test_result = None
    coverage = 0

    batch_size=args.ssl_batch_size
    n_users=60742
    n_batch = (n_users + batch_size - 1) // batch_size
    
    start_time = time.time()
    print_interval = 6  

    for epoch in range(args.epoch):
        model.train()
        dst, graph_neg = construct_negative_graph(graph,('user','play','game'),device)
        h,h_sub1,h_sub2 = model()
        ssloss_value=0

        for idx in range(n_batch):

            ua_embeddings_sub1,ia_embeddings_sub1=h_sub1['user'],h_sub1['game']
            ua_embeddings_sub2,ia_embeddings_sub2=h_sub2['user'],h_sub2['game']            
            start_idx = idx * batch_size
            end_idx = min(start_idx + batch_size, n_users)

            batch_ua_embeddings_sub1 = ua_embeddings_sub1[start_idx:end_idx]
            batch_ia_embeddings_sub1 = ia_embeddings_sub1[start_idx:end_idx]
            batch_ua_embeddings_sub2 = ua_embeddings_sub2[start_idx:end_idx]
            batch_ia_embeddings_sub2 = ia_embeddings_sub2[start_idx:end_idx]

            ssloss_value =ssloss_value + ssloss.forward(batch_ua_embeddings_sub1, batch_ua_embeddings_sub2, batch_ia_embeddings_sub1, batch_ia_embeddings_sub2,device)

        score = predictor(graph, h, ('user','play','game'))
        score_neg = predictor(graph_neg, h, ('user','play','game'))
        loss_pre = loss
        
        eps=1e-15
        score_neg_reweight = score_neg * (1 / (1 + torch.exp(-score_neg*args.balance)) * args.K)
        loss =  (-((score - score_neg_reweight).sigmoid().clamp(min=1e-8, max=1-1e-8).log())).sum()

        loss = loss.to(device)
        loss_ind_cons = orthogonal_loss(model.preference_embedding)
        total_loss=loss + ssloss_value*args.ssl_loss_weight + loss_ind_cons
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        total_epoch += 1
        
        to_get_coverage = True
        epoch_inter = 50
        if total_epoch > 0 and total_epoch % epoch_inter == 0:
            logging.info("\n"+"="*40)
            logging.info('Epoch {}'.format(epoch))
            logging.info(f"loss = {loss}")
            model.eval()
            logging.info("begin validation")

            _, result = validate(valid_mask, DataLoader.valid_data, h, ls_k, DataLoader.game_genre_mapping,to_get_coverage,device)

            if loss < loss_pre:
                stop_count = 0
                logging.info("begin test")
                _, test_result = validate(valid_mask, DataLoader.test_data, h, ls_k, DataLoader.game_genre_mapping, to_get_coverage,device)
            else:
                stop_count += 1
                logging.info(f"stop count:{stop_count}")
                if stop_count > args.early_stop:
                    logging.info('early stop')
                    break
 
    logging.info(test_result)


