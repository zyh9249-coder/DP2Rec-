import os
import sys
import numpy as np
import torch
import dgl
from tqdm import tqdm
import pickle
import logging
from collections import defaultdict, Counter
import random
import multiprocessing as mp
import threading
import os
import time
import sys
import csv
from parser_div import parse_args
from dataloader_steam import Dataloader_steam_filtered
import multiprocessing as mp
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device


DEVICE = get_device()


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cudnn.benchmark = True

def min_max_normalize(values):
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [1.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def roulette_wheel_selection(probabilities):
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return i
    return len(probabilities) - 1  

def prepare_embeddings_gpu(embeddings_dict):
    ids = []
    embeddings = []
    
    for game_id, embedding in embeddings_dict.items():
        ids.append(game_id)
        embeddings.append(embedding)
    
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    id_to_idx = {game_id: i for i, game_id in enumerate(ids)}
    idx_to_id = {i: game_id for i, game_id in enumerate(ids)}

    return embeddings_tensor, id_to_idx, idx_to_id

def precompute_genre_similar_games_XR(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games, cache_path = ''):
    
    # ===== 1️⃣ 读取缓存 =====
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            precomputed_similar_games = pickle.load(f)
        logger.info(f"Loaded from cache: {cache_path}, covering {len(precomputed_similar_games)} games")
        return precomputed_similar_games

    # ===== 2️⃣ 预处理 =====
    precomputed_similar_games = {}
    all_game_ids = list(id_to_idx.keys())
    all_indices = torch.arange(len(all_game_ids))
    
    # 👉 可选：做归一化（强烈推荐，用于 cosine similarity）
    embeddings = embeddings_tensor
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

    # ===== 3️⃣ 主循环 =====
    for game_id in tqdm(all_game_ids):
        if game_id not in id_to_idx:
            continue

        current_idx = id_to_idx[game_id]
        current_emb = embeddings[current_idx].unsqueeze(0)  # (1, d)

        # ===== 计算和所有游戏的相似度 =====
        sims = torch.mm(current_emb, embeddings.t()).squeeze(0)  # (N,)

        # 排除自己
        sims[current_idx] = -1e9

        # ===== 取 Top-K =====
        topk_sim, topk_idx = torch.topk(sims, 20)

        game_similar_games = []

        for sim, idx in zip(topk_sim.tolist(), topk_idx.tolist()):
            best_game_id = all_game_ids[idx]

            # 👉 取一个 genre（可以取第一个 or 随机 or 多个）
            genres = game_to_genres.get(best_game_id, [])
            genre_id = genres[0] if len(genres) > 0 else None

            game_similar_games.append((best_game_id, sim, genre_id))

        precomputed_similar_games[game_id] = game_similar_games

    logger.info(f"Successfully computed Top-{20} similar games for {len(precomputed_similar_games)} games.")

    # ===== 4️⃣ 保存缓存 =====
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(precomputed_similar_games, f)

    return precomputed_similar_games

def precompute_genre_similar_games(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games, cache_path = ''):
    
    # 如果缓存文件已存在，直接读取
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            precomputed_similar_games = pickle.load(f)
        logger.info(f"Loaded genre_similar_games from cache: {cache_path}, covering {len(precomputed_similar_games)} games")
        return precomputed_similar_games
    
    precomputed_similar_games = {}
    all_game_ids = list(id_to_idx.keys())
    
    for game_id in tqdm(all_game_ids):
        if game_id not in id_to_idx:
            continue
        current_idx = id_to_idx[game_id]
        
        all_genres = list(genre_to_games.keys())
        
        game_similar_games = []
        
        for genre_id in all_genres:
            genre_games = genre_to_games[genre_id]
            genre_games = [g for g in genre_games if g != game_id and g in id_to_idx]
            if not genre_games:
                continue
            genre_indices = [id_to_idx[g] for g in genre_games]
            
            current_embedding = embeddings_tensor[current_idx].unsqueeze(0)
            genre_embeddings = embeddings_tensor[genre_indices]
            
            modal_similarities = torch.mm(current_embedding, genre_embeddings.t()).squeeze(0)
            if len(modal_similarities) > 0:
                best_idx = torch.argmax(modal_similarities).item()
                best_game_id = genre_games[best_idx]
                best_similarity =modal_similarities[best_idx].item()
                game_similar_games.append((best_game_id, best_similarity, genre_id))
        
        precomputed_similar_games[game_id] = game_similar_games
    
    logger.info(f"Successfully precomputed {len(precomputed_similar_games)} games' similar games in each category.")
    
    # 保存成 pkl 文件，方便下次直接加载
    with open(cache_path, "wb") as f:
        pickle.dump(precomputed_similar_games, f)

    return precomputed_similar_games

def reverse_mapping(mapping):
    return {v: k for k, v in mapping.items()}

def load_game_embeddings(embedding_folder):
    embeddings = {}
    files = os.listdir(embedding_folder)
    for file in tqdm(files):
        if file.endswith('.npy'):
            game_id = file.split('.')[0]
            embedding = np.load(os.path.join(embedding_folder, file))
            embeddings[game_id] = embedding.flatten() 

    logger.info(f"Successfully loaded embeddings for {len(embeddings)} games.")
    return embeddings

def load_user_game_interactions(DataLoader):
    dic_user_games_set = {}
    dic_user_game_time = {}
    
    dic_user_games = DataLoader.dic_user_games
    tensor_user_game = DataLoader.user_game_time

    for i in range(tensor_user_game.shape[0]):
        user_id = int(tensor_user_game[i, 0].item())
        game_id = int(tensor_user_game[i, 1].item())
        time_percentile = tensor_user_game[i, 3].item()  

        if user_id not in dic_user_game_time:
            dic_user_game_time[user_id] = {}

        dic_user_game_time[user_id][game_id] = time_percentile

    for user_id, games in dic_user_games.items():
        dic_user_games_set[user_id] = set(games)

    logger.info(f"Successfully loaded game interaction information for {len(dic_user_games_set)} users.")
        
    return dic_user_games_set, dic_user_game_time

def get_S_user_games(S_graph, user_id):

    src, dst = S_graph.out_edges(user_id, etype='play')
    if len(src) == 0:
        return []
    
    S_games = dst.tolist()
    return S_games


def find_similar_games_per_genre(current_game_id, precomputed_similar_games):
    return precomputed_similar_games.get(current_game_id, [])

def load_game_time_similarity(file_path, app_id_forward, cache_path="game_time_sim.pkl"):
    # 如果缓存文件已存在，直接读取
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            game_time_sim = pickle.load(f)
        logger.info(f"Loaded game_time_sim from cache: {cache_path}, covering {len(game_time_sim)} games")
        return game_time_sim

    # 否则按原来的方式加载
    game_time_sim = {}
    int_to_numeric_id = {}
    for orig_id, numeric_id in app_id_forward.items():
        try:
            int_id = int(orig_id)
            int_to_numeric_id[int_id] = numeric_id
        except ValueError:
            pass

    count = 0
    matched = 0
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in tqdm(reader, desc="Processing similarity data"):
            if len(row) < 3:
                continue
                
            try:
                game1_int = int(row[0])
                game2_int = int(row[1])
                similarity = float(row[2])
                
                count += 1
                
                if game1_int in int_to_numeric_id and game2_int in int_to_numeric_id:
                    game1 = int_to_numeric_id[game1_int]
                    game2 = int_to_numeric_id[game2_int]
                    
                    if game1 not in game_time_sim:
                        game_time_sim[game1] = {}
                    if game2 not in game_time_sim:
                        game_time_sim[game2] = {}
                        
                    game_time_sim[game1][game2] = similarity
                    game_time_sim[game2][game1] = similarity
                    
                    matched += 1
            except (ValueError, IndexError):
                continue
                
            if count % 1000000 == 0:
                logger.info(f"Processed {count} game pairs, successfully matched {matched} pairs")
    
    logger.info(f"Total processed {count} game pairs, successfully matched {matched} pairs")
    logger.info(f"Successfully loaded game time similarity data, covering {len(game_time_sim)} games")

    # 保存成 pkl 文件，方便下次直接加载
    with open(cache_path, "wb") as f:
        pickle.dump(game_time_sim, f)

    return game_time_sim

def process_user(user_id, dn_graph, dic_user_games_set, dic_user_game_time, 
                precomputed_similar_games, game_to_genres, game_time_sim, max_per_genre):

    original_games = dic_user_games_set.get(user_id, set())
    denoised_games = get_S_user_games(dn_graph, user_id)

    if not denoised_games:
        return [], []
    
    game_times = {}
    for game_id in denoised_games:
        if user_id in dic_user_game_time and game_id in dic_user_game_time[user_id]:
            game_times[game_id] = dic_user_game_time[user_id][game_id]
    
    if not game_times:
        return [], []
    
    added_games = set()
    
    user_to_game_edges = []
    user_game_weights = []

    game_level_info = {}  
    explored_initial_nodes = set()  
    exploration_results = {}  
    
    genre_counts = defaultdict(int)
    count = 0
    while True:
        count += 1
        if count == 200:
            break
        
        if not game_times:
            break
            
        times = list(game_times.values())
        game_ids = list(game_times.keys())
        total_time = sum(times)
        
        if total_time == 0:
            probs = [1/len(times)] * len(times)
        else:
            probs = [t/total_time for t in times]
            
        selected_idx = roulette_wheel_selection(probs)
        initial_game_id = game_ids[selected_idx]
        
        if initial_game_id in explored_initial_nodes:
            previous_added = exploration_results.get(initial_game_id, [])
            if not previous_added:
                continue
            start_game_id = random.choice(previous_added)
        else:
            start_game_id = initial_game_id
            game_level_info[initial_game_id] = (1, [initial_game_id], 1.0)
            exploration_results[initial_game_id] = []
            explored_initial_nodes.add(initial_game_id)
        
        current_level, current_path, current_accumulated_sim = game_level_info.get(start_game_id, (1, [start_game_id], 1.0))
        # similar_games = find_similar_games_per_genre(start_game_id, precomputed_similar_games)
        similar_games = precomputed_similar_games.get(start_game_id, [])
        
        filtered_similar_games = []
        for game_id, modal_sim, genre_id in similar_games:
            if game_id not in original_games and game_id not in added_games:
                time_sim = 0.0
                if start_game_id in game_time_sim and game_id in game_time_sim[start_game_id]:
                    time_sim = game_time_sim[start_game_id][game_id]
                filtered_similar_games.append((game_id, modal_sim, time_sim, genre_id))
        
        if not filtered_similar_games:
            continue
    
        candidates = []
        sim_values = []
        time_sim_values = []
        category_values = []
        
        for game_id, modal_sim, time_sim, genre_id in filtered_similar_games:
            current_count = genre_counts[genre_id]
            
            if current_count >= max_per_genre:
                continue
            
            remaining_ratio = (max_per_genre - current_count) / max_per_genre
            
            candidates.append((game_id, genre_id))
            sim_values.append(modal_sim)
            time_sim_values.append(time_sim)
            category_values.append(remaining_ratio)
        
        if not candidates:
            continue
        norm_sim_values = min_max_normalize(sim_values)
        norm_time_sim_values = min_max_normalize(time_sim_values)
        norm_category_values = min_max_normalize(category_values)
        
        scores = []
        for i in range(len(candidates)):
            score = (norm_sim_values[i] + norm_time_sim_values[i] + norm_category_values[i]) / 3
            # score = (norm_sim_values[i] + norm_time_sim_values[i]) / 2
            scores.append(score)
        
        total_score = sum(scores)
        if total_score == 0:
            probs = [1/len(scores)] * len(scores)
        else:
            probs = [s/total_score for s in scores]
        
        selected_idx = roulette_wheel_selection(probs)
        selected_game_id, selected_genre_id = candidates[selected_idx]
        
        user_to_game_edges.append((user_id, selected_game_id))
        selected_genre_count = len(game_to_genres.get(selected_game_id, []))

        modal_sim = sim_values[selected_idx]
        
        new_level = current_level + 1
        new_path = current_path + [selected_game_id]
        new_accumulated_sim = current_accumulated_sim * modal_sim
        
        game_level_info[selected_game_id] = (new_level, new_path, new_accumulated_sim)
        
        if new_level == 1: 
            user_game_time = dic_user_game_time[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time
        elif new_level == 2:  
            initial_game_id = new_path[0]
            user_game_time = dic_user_game_time[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time * modal_sim
        else:  
            initial_game_id = new_path[0]
            user_game_time = dic_user_game_time[user_id].get(initial_game_id, 0)
            edge_weight = user_game_time * new_accumulated_sim
        
        user_game_weights.append(edge_weight)
        
        added_games.add(selected_game_id)
        
        for genre_id in game_to_genres.get(selected_game_id, []):
            genre_counts[genre_id] += 1
        
        if initial_game_id in exploration_results:
            exploration_results[initial_game_id].append(selected_game_id)
            
        all_full = True
        for genre_id in genre_counts.keys():
            if genre_counts[genre_id] < max_per_genre:
                all_full = False
                break
        
        if all_full :
            break
        # if count == 15:
        #     break
    
    return user_to_game_edges, user_game_weights

import random
from collections import defaultdict


def process_user_CBC(
    user_id,
    dn_graph,
    dic_user_games_set,
    dic_user_game_time,
    precomputed_similar_games,
    game_time_sim,
    max_walk_steps=100,
):
    """
    对单个用户进行随机游走扩展，生成新的 user-game 边及其权重。

    参数说明
    ----------
    user_id : int / str
        当前用户 ID。

    dn_graph :
        去噪后的图，用于获取用户去噪后的游戏集合。

    dic_user_games_set : dict
        用户原始玩过的游戏集合。
        形式：
        {
            user_id: {game_id1, game_id2, ...}
        }

    dic_user_game_time : dict
        用户对游戏的游玩时长。
        形式：
        {
            user_id: {
                game_id: play_time,
                ...
            }
        }

    precomputed_similar_games : dict
        预先计算好的相似游戏。
        形式：
        {
            game_id: [
                (similar_game_id, modal_sim, genre_id),
                ...
            ]
        }

    game_time_sim : dict
        游戏之间的时间相似度。
        形式：
        {
            game_id1: {
                game_id2: time_sim,
                ...
            }
        }

    max_walk_steps : int
        最大随机游走次数，防止死循环。

    返回
    ----------
    user_to_game_edges : list
        新增的 user-game 边。
        形式：
        [(user_id, selected_game_id), ...]

    user_game_weights : list
        每条新增边对应的权重。
        形式：
        [weight1, weight2, ...]
    """

    # =========================
    # 1. 获取用户原始游戏和去噪后的游戏
    # =========================
    original_games = dic_user_games_set.get(user_id, set())
    denoised_games = get_S_user_games(dn_graph, user_id)

    if not denoised_games:
        return [], []

    # =========================
    # 2. 只保留有游玩时长记录的去噪游戏
    # =========================
    game_times = {}

    if user_id not in dic_user_game_time:
        return [], []

    for game_id in denoised_games:
        if game_id in dic_user_game_time[user_id]:
            game_times[game_id] = dic_user_game_time[user_id][game_id]

    if not game_times:
        return [], []

    # =========================
    # 3. 初始化结果变量
    # =========================
    added_games = set()

    user_to_game_edges = []
    user_game_weights = []

    # 记录每个游戏的层级、路径、路径累计相似度
    # game_id -> (level, path, accumulated_sim)
    game_level_info = {}

    # 记录哪些原始游戏已经被探索过
    explored_initial_nodes = set()

    # 记录每个原始游戏扩展出来的游戏
    # initial_game_id -> [added_game1, added_game2, ...]
    exploration_results = {}

    # =========================
    # 4. 开始随机游走
    # =========================
    for step in range(max_walk_steps):

        if not game_times:
            break

        # =========================
        # 4.1 根据用户游玩时长选择一个初始游戏
        # =========================
        game_ids = list(game_times.keys())
        times = list(game_times.values())
        total_time = sum(times)

        if total_time == 0:
            probs = [1.0 / len(times)] * len(times)
        else:
            probs = [t / total_time for t in times]

        selected_idx = roulette_wheel_selection(probs)
        initial_game_id = game_ids[selected_idx]

        # =========================
        # 4.2 确定本轮随机游走的起点 start_game_id
        # =========================
        if initial_game_id in explored_initial_nodes:
            previous_added = exploration_results.get(initial_game_id, [])

            # 如果这个初始游戏之前没有成功扩展出游戏，则跳过
            if not previous_added:
                continue

            # 从之前扩展出来的游戏中随机选一个继续游走
            start_game_id = random.choice(previous_added)

        else:
            # 第一次探索这个初始游戏，则从它自身开始
            start_game_id = initial_game_id

            game_level_info[initial_game_id] = (
                1,
                [initial_game_id],
                1.0
            )

            exploration_results[initial_game_id] = []
            explored_initial_nodes.add(initial_game_id)

        # =========================
        # 4.3 获取当前游走起点的层级、路径、累计相似度
        # =========================
        current_level, current_path, current_accumulated_sim = game_level_info.get(
            start_game_id,
            (1, [start_game_id], 1.0)
        )

        # =========================
        # 4.4 获取 start_game_id 的相似游戏
        # =========================
        similar_games = precomputed_similar_games.get(start_game_id, [])

        if not similar_games:
            continue

        # =========================
        # 4.5 过滤候选游戏
        # =========================
        candidates = []
        sim_values = []
        time_sim_values = []

        for candidate_game_id, modal_sim, genre_id in similar_games:

            # 不添加用户原本玩过的游戏
            if candidate_game_id in original_games:
                continue

            # 不重复添加已经扩展过的游戏
            if candidate_game_id in added_games:
                continue

            # 获取时间相似度
            time_sim = 0.0
            if (
                start_game_id in game_time_sim
                and candidate_game_id in game_time_sim[start_game_id]
            ):
                time_sim = game_time_sim[start_game_id][candidate_game_id]

            candidates.append(candidate_game_id)
            sim_values.append(modal_sim)
            time_sim_values.append(time_sim)

        if not candidates:
            continue

        # =========================
        # 4.6 对相似度进行归一化
        # =========================
        norm_sim_values = min_max_normalize(sim_values)
        norm_time_sim_values = min_max_normalize(time_sim_values)

        # =========================
        # 4.7 计算候选游戏采样分数
        # =========================
        scores = []

        for i in range(len(candidates)):
            score = (
                norm_sim_values[i]
                + norm_time_sim_values[i]
            ) / 2.0

            scores.append(score)

        # 如果所有分数都是 0，则等概率随机选择
        total_score = sum(scores)

        if total_score == 0:
            candidate_probs = [1.0 / len(scores)] * len(scores)
        else:
            candidate_probs = [s / total_score for s in scores]

        # =========================
        # 4.8 随机选择一个候选游戏
        # =========================
        selected_candidate_idx = roulette_wheel_selection(candidate_probs)

        selected_game_id = candidates[selected_candidate_idx]
        selected_modal_sim = sim_values[selected_candidate_idx]

        # =========================
        # 4.9 更新路径信息
        # =========================
        new_level = current_level + 1
        new_path = current_path + [selected_game_id]
        new_accumulated_sim = current_accumulated_sim * selected_modal_sim

        game_level_info[selected_game_id] = (
            new_level,
            new_path,
            new_accumulated_sim
        )

        # =========================
        # 4.10 计算 user-game 边权重
        # =========================
        root_game_id = new_path[0]
        user_game_time = dic_user_game_time[user_id].get(root_game_id, 0)

        edge_weight = user_game_time * new_accumulated_sim

        # =========================
        # 4.11 保存新增边和边权重
        # =========================
        user_to_game_edges.append((user_id, selected_game_id))
        user_game_weights.append(edge_weight)

        # =========================
        # 4.12 更新已添加游戏集合
        # =========================
        added_games.add(selected_game_id)

        # =========================
        # 4.13 记录该游戏是从哪个初始游戏扩展出来的
        # =========================
        if root_game_id in exploration_results:
            exploration_results[root_game_id].append(selected_game_id)

    return user_to_game_edges, user_game_weights


def batch_process_users(user_ids, dn_graph, dic_user_games_set, dic_user_game_time, 
                     precomputed_similar_games, game_to_genres, game_time_sim, max_per_genre, batch_size=1024):

    all_edges = []
    all_weights = []
    
    num_batches = (len(user_ids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(user_ids))
        batch_user_ids = user_ids[start_idx:end_idx]
        
        batch_results = []
        for user_id in batch_user_ids:

            start_time = time.time()
            edges, weights = process_user(
                user_id, 
                dn_graph, 
                dic_user_games_set, 
                dic_user_game_time, 
                precomputed_similar_games,
                game_to_genres,
                game_time_sim,
                max_per_genre
            )
            # edges, weights = process_user_CBC(
            #     user_id,
            #     dn_graph=dn_graph,
            #     dic_user_games_set=dic_user_games_set,
            #     dic_user_game_time=dic_user_game_time,
            #     precomputed_similar_games=precomputed_similar_games,
            #     game_time_sim=game_time_sim,
            #     max_walk_steps=10,
            # )

            batch_results.append((edges, weights))

        for edges, weights in batch_results:
            all_edges.extend(edges)
            all_weights.extend(weights)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_edges, all_weights

def main():
    set_random_seed(2025)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    args = parse_args()
    path = "steam_data"
    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genres_path = path + '/Games_Genres.txt'
    embedding_folder = "steam_data/modal_embeddings"   
    S_graph_path ="data_exist/old_genre/S_graph.bin"
    
    if 'init' in genres_path:
        output_graph_path = "data_exist/mrw.bin" 
        weights_path = "data_exist/weights_mrw.pth"
    else:
        output_graph_path = "data_exist/old_genre/mrw.bin" 
        weights_path = "data_exist/old_genre/weights_mrw.pth"
        
    # if os.path.exists(output_graph_path) and os.path.exists(weights_path):
    #     logger.info(f"Output files {output_graph_path} and {weights_path} already exist, skipping entire processing pipeline")
    #     return
    logger.info("Loading data...")
    DataLoader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, genres_path)
    
    logger.info("Creating ID mappings...")
    app_id_reverse = reverse_mapping(DataLoader.app_id_mapping)
    app_id_forward = DataLoader.app_id_mapping  
    user_id_reverse = reverse_mapping(DataLoader.user_id_mapping)
    user_id_forward = DataLoader.user_id_mapping  

    game_time_sim_path = "data_exist/game_similarity.csv"
    game_time_sim_pkl_path = "data_exist/game_time_sim.pkl"
    game_time_sim = load_game_time_similarity(game_time_sim_path, app_id_forward, game_time_sim_pkl_path)

    logger.info("Loading genre mappings...")
    genre_id_to_name = {}
    name_to_genre_id = {}
    with open(genres_path, 'r') as f:
        genre_set = set()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[1]:
                genre_set.add(parts[1])
        
        for i, genre in enumerate(sorted(genre_set)):
            genre_id_to_name[i] = genre
            name_to_genre_id[genre] = i

    game_to_genres = {}
    game_to_genre_names = {}
    for game_id, genre_ids in DataLoader.game_genre_mapping.items():
        game_to_genres[game_id] = genre_ids
        game_to_genre_names[game_id] = [genre_id_to_name.get(genre_id, f"类别_{genre_id}") for genre_id in genre_ids]

    genre_to_games = defaultdict(list)
    for game_id, genre_ids in game_to_genres.items():
        for genre_id in genre_ids:
            genre_to_games[genre_id].append(game_id)
    
    original_game_embeddings = load_game_embeddings(embedding_folder)
    embeddings_mapped = {}
    
    for original_id, embedding in original_game_embeddings.items():
        if original_id in app_id_forward:
            numeric_id = app_id_forward[original_id]
            embeddings_mapped[numeric_id] = embedding
            
    embeddings_tensor, id_to_idx, idx_to_id = prepare_embeddings_gpu(embeddings_mapped)

    if 'init' in genres_path:
        genre_similar_games_pkl_path = '/mnt/data/zhangyuhang/Recommender_System/DP^2Rec_init/data_exist/genre_similar_games.pkl'
    else:
        # genre_similar_games_pkl_path = '/mnt/data/zhangyuhang/Recommender_System/DP^2Rec_init/data_exist/old_genre/genre_similar_games.pkl'
        genre_similar_games_pkl_path = '/mnt/data/zhangyuhang/Recommender_System/DP^2Rec_init/data_exist/old_genre/genre_similar_games_XR.pkl'
    # {game_id : [(best_game_id, best_similarity, genre_id)]...}  返回相似度字典
    # precomputed_similar_games = precompute_genre_similar_games(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games, genre_similar_games_pkl_path)
    precomputed_similar_games = precompute_genre_similar_games_XR(embeddings_tensor, id_to_idx, game_to_genres, genre_to_games, genre_similar_games_pkl_path)
    dic_user_games_set, dic_user_game_time = load_user_game_interactions(DataLoader)
    
    dn_graph, _ = dgl.load_graphs(S_graph_path)
    dn_graph = dn_graph[0]

    if torch.cuda.is_available():
        dn_graph = dn_graph.to(DEVICE)

    test_users = list(DataLoader.test_data.keys())
    
    batch_size = 1
    all_edges, all_weights = batch_process_users(
        test_users, 
        dn_graph, 
        dic_user_games_set, 
        dic_user_game_time, 
        precomputed_similar_games,
        game_to_genres,
        game_time_sim,
        args.max_per_genre,  
        batch_size
    )
    
    
    src_nodes = torch.tensor([edge[0] for edge in all_edges], dtype=torch.int64)
    dst_nodes = torch.tensor([edge[1] for edge in all_edges], dtype=torch.int64)
    
    edge_weights = torch.tensor(all_weights, dtype=torch.float32)
    
    actual_users = torch.unique(src_nodes).shape[0]
    actual_games = torch.unique(dst_nodes).shape[0]
    
    logger.info(f"Number of users with actual connections: {actual_users}")
    logger.info(f"Number of games with actual connections: {actual_games}")

    graph_data = {
        ('user', 'plays', 'game'): (src_nodes, dst_nodes),
        ('game', 'played_by', 'user'): (dst_nodes, src_nodes)
    }
    
    num_nodes_dict = {
        'user': DataLoader.graph.number_of_nodes('user'),
        'game': DataLoader.graph.number_of_nodes('game')
    }
    
    diversity_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    
    diversity_graph.edges['plays'].data['weight'] = edge_weights
    diversity_graph.edges['played_by'].data['weight'] = edge_weights
    
    logger.info(f"Saving diversity exploration graph to {output_graph_path}")

    dgl.save_graphs(output_graph_path, [diversity_graph])
    
    torch.save(edge_weights, weights_path)
    logger.info(f"Edge weights saved to {weights_path}")

    
    num_users = diversity_graph.number_of_nodes('user')
    num_games = diversity_graph.number_of_nodes('game')
    num_edges = diversity_graph.number_of_edges('plays')

    logger.info("Diversity exploration graph statistics:")
    logger.info(f"  - Total number of users: {num_users}")
    logger.info(f"  - Total number of games: {num_games}")
    logger.info(f"  - Number of edges: {num_edges}")

    avg_connections = num_edges / actual_users if actual_users > 0 else 0
    logger.info(f"  - Average number of games per active user: {avg_connections:.2f}")

    user_genre_coverage = defaultdict(set)

    for i in range(len(all_edges)):
        user_id, game_id = all_edges[i]
        for genre_id in game_to_genres.get(game_id, []):
            user_genre_coverage[user_id].add(genre_id)

    coverage_counts = [len(genres) for user_id, genres in user_genre_coverage.items()]

    if coverage_counts:
        avg_genre_coverage = sum(coverage_counts) / len(coverage_counts)
        max_genre_coverage = max(coverage_counts)
        min_genre_coverage = min(coverage_counts)

        logger.info(f"  - Average number of genres covered per user: {avg_genre_coverage:.2f}")
        logger.info(f"  - Maximum genre coverage: {max_genre_coverage}")
        logger.info(f"  - Minimum genre coverage: {min_genre_coverage}")

    logger.info("Calculating connection statistics per genre...")
    genre_connected_users = defaultdict(set)
    genre_interaction_count = defaultdict(int)

    for i in range(len(all_edges)):
        user_id, game_id = all_edges[i]
        for genre_id in game_to_genres.get(game_id, []):
            genre_connected_users[genre_id].add(user_id)
            genre_interaction_count[genre_id] += 1

    logger.info("Connection statistics per game genre:")
    sorted_genres = sorted([(genre_id, genre_interaction_count[genre_id], len(genre_connected_users[genre_id])) 
                            for genre_id in genre_interaction_count.keys()], 
                        key=lambda x: x[1], reverse=True)

    for genre_id, interaction_count, user_count in sorted_genres:
        genre_name = genre_id_to_name.get(genre_id, f"Genre_{genre_id}")
        total_genre_games = len(genre_to_games[genre_id])
        avg_interactions_per_user = interaction_count / user_count if user_count > 0 else 0
        logger.info(f"  - {genre_name}: {interaction_count} total interactions, {user_count} users, "
                    f"avg {avg_interactions_per_user:.2f} games per user in this genre "
                    f"(total {total_genre_games} games in this genre)")

    logger.info("Diversity exploration graph construction completed!")

if __name__ == "__main__":
    main()