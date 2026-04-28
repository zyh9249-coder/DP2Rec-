import os
import sys  
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats

import pdb
import torch
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
from dgl.data import DGLDataset
import pickle


class Dataloader_item_graph(DGLDataset):
    def __init__(self, app_id_path, genre_path, dataloader_steam):
        self.app_id_path = app_id_path
        self.genre_path = genre_path

        logging.info("reading item graph...")
        self.app_id_mapping = dataloader_steam.app_id_mapping
        self.genre = dataloader_steam.game_genre_mapping

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        # path_dic_genre = os.path.join(base_dir, "data_exist/dic_genre.pkl")

        # if not os.path.exists(path_dic_genre) :
        #     with open(path_dic_genre, 'wb') as f:
        #         pickle.dump(self.genre, f)
        
        # 同类别下的游戏中间有边
        path_graph_item = os.path.join(base_dir, "data_exist/graph_item.bin")
        
        if os.path.exists(path_graph_item):
            self.graph_item,_ = dgl.load_graphs(path_graph_item)
            self.graph_item = self.graph_item[0]
        else:
            self.genre_item = self.build_edge_item(self.genre)
            graph_data_item = {
                ('game', 'co_genre', 'game'): self.genre_item,
            }
            graph_item=dgl.heterograph(graph_data_item)

            self.graph_item=graph_item
            dgl.save_graphs(path_graph_item,[self.graph_item])

        



    def build_edge_item(self, mapping):
        src = []
        dst = []
        keys = list(set(mapping.keys()))
        for game in keys:
            mapping[game] = set(mapping[game])

        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                if len(mapping[game1] & mapping[game2]) > 0:  
                    src.extend([game1, game2])  
                    dst.extend([game2, game1])  
        return (torch.tensor(src), torch.tensor(dst))
    
    def build_edge_or(self, mapping1, mapping2, mapping3):
        src = []
        dst = []
        keys = list(set(mapping1.keys()) | set(mapping2.keys()) | set(mapping3.keys()))

        for game in keys:
            if game in mapping1:
                mapping1[game] = set(mapping1[game])
            else:
                mapping1[game] = set()
            if game in mapping2:
                mapping2[game] = set(mapping2[game])
            else:
                mapping2[game] = set()
            if game in mapping3:
                mapping3[game] = set(mapping3[game])
            else:
                mapping3[game] = set()

        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)): 
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping1[game1]) & set(mapping1[game2])) > 0 or len(set(mapping2[game1]) & set(mapping2[game2])) > 0 or len(set(mapping3[game1]) & set(mapping3[game2])) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))
    
if __name__ == "__main__":
    Dataloader_item_graph()
