import os
import sys
import torch
import dgl
import numpy as np
from tqdm import tqdm
import logging
import pickle
from collections import defaultdict
import csv
import multiprocessing as mp
from itertools import combinations
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_game_pairs(game_pairs, game_to_users, user_to_games, reverse_app_id_mapping, queue):

    results = []
    
    for game_i, game_j in game_pairs:
        if game_i not in reverse_app_id_mapping or game_j not in reverse_app_id_mapping:
            continue
            
        orig_game_i = reverse_app_id_mapping[game_i]
        orig_game_j = reverse_app_id_mapping[game_j]
        
        users_i = set(game_to_users[game_i])
        users_j = set(game_to_users[game_j])
        common_users = users_i.intersection(users_j)
        common_users_count = len(common_users)
        
        if common_users_count > 0:

            total_diff = 0.0
            for user_id in common_users:
                posterior_i = user_to_games[user_id].get(game_i, 0.0)
                posterior_j = user_to_games[user_id].get(game_j, 0.0)
                total_diff += abs(posterior_i - posterior_j)
            
            similarity = 1.0 - (total_diff / common_users_count)
            
            results.append((orig_game_i, orig_game_j, similarity, common_users_count))
    
    queue.put(results)

def calculate_game_similarity(graph_path, app_id_mapping_path, output_csv_path, batch_size=10000, num_processes=None):
    start_time = time.time()
    
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    

    with open(app_id_mapping_path, 'rb') as f:
        app_id_mapping = pickle.load(f)
    
    reverse_app_id_mapping = {v: k for k, v in app_id_mapping.items()}
    
    logger.info(f"Loading graph data from: {graph_path}")
    graph, _ = dgl.load_graphs(graph_path)
    graph = graph[0]

    
    user_game_edges = graph.edges(etype='play')
    game_user_edges = graph.edges(etype='played by')
    user_game_posteriors = graph.edges['play'].data['em_posterior']
    
    user_to_games = defaultdict(dict)  
    game_to_users = defaultdict(list)  
    
    for i in tqdm(range(len(user_game_edges[0]))):
        user_id = user_game_edges[0][i].item()
        game_id = user_game_edges[1][i].item()
        posterior = user_game_posteriors[i].item()
        user_to_games[user_id][game_id] = posterior
    
    for i in tqdm(range(len(game_user_edges[0]))):
        game_id = game_user_edges[0][i].item()
        user_id = game_user_edges[1][i].item()
        game_to_users[game_id].append(user_id)
    
    del graph, user_game_edges, game_user_edges, user_game_posteriors
    
    all_games = sorted(list(game_to_users.keys()))
    num_games = len(all_games) # 7717 说明不是所有游戏都和用户有交互（训练集）

    
    all_game_pairs = list(combinations(all_games, 2))
    total_pairs = len(all_game_pairs)
    
    batches = [all_game_pairs[i:i+batch_size] for i in range(0, total_pairs, batch_size)]
    num_batches = len(batches)
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Game 1', 'Game 2', 'Similarity'])

        
        pairs_with_common_users = 0
        processed_batches = 0
        
        pbar = tqdm(total=num_batches)
        
        for i in range(0, len(batches), num_processes):
            current_batches = batches[i:i+num_processes]
            
            processes = []
            for batch in current_batches:
                p = mp.Process(target=process_game_pairs, args=(batch, game_to_users, user_to_games, reverse_app_id_mapping, result_queue))
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            batch_results = []
            for _ in range(len(current_batches)):
                batch_results.extend(result_queue.get())
            
            write_pbar = tqdm(total=len(batch_results))
            
            for row in batch_results:
                writer.writerow(row[:3])
                write_pbar.update(1)
            
            write_pbar.close()
            f.flush()  
            
            pairs_with_common_users += len(batch_results)
            processed_batches += len(current_batches)
            
            pbar.update(len(current_batches))
            
            elapsed_time = time.time() - start_time
            completion = processed_batches / num_batches
            estimated_total = elapsed_time / completion if completion > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            logger.info(f"Processed {processed_batches}/{num_batches} batches ({completion*100:.2f}%)")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds, Estimated remaining time: {remaining_time:.2f} seconds")

        
        pbar.close()
    
    logger.info(f"Processing complete: Calculated similarity for {total_pairs} game pairs")
    logger.info(f"Game pairs with common users: {pairs_with_common_users} ({pairs_with_common_users / total_pairs * 100:.2f}%)")
    logger.info(f"Similarity data saved to: {output_csv_path}")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")


def main():
    graph_path = "data_exist/graph.bin"
    app_id_mapping_path = "data_exist/app_id_mapping.pkl"
    output_csv_path = "data_exist/game_similarity.csv"
    # if os.path.exists(output_csv_path):
    #     logger.info(f"Output file {output_csv_path} already exists, skipping entire processing pipeline")
    #     return
    num_processes = max(1, mp.cpu_count() - 1)
    batch_size = 10000  
    
    calculate_game_similarity(graph_path, app_id_mapping_path, output_csv_path, 
                            batch_size=batch_size, num_processes=num_processes)

if __name__ == "__main__":
    main()