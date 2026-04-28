import torch
import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import dgl
from dataloader_steam import Dataloader_steam_filtered
from parser_acc import parse_args

def main():

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    path = "./steam_data"
    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    genre_path = path + '/Games_Genres_init.txt'
    if "init" in genre_path: 
        save_dir = './data_exist'
        S_graph_path = save_dir + "/S_graph.bin"
    else:
        save_dir = './data_exist/old_genre'
        S_graph_path = save_dir + "/old_genre/S_graph.bin"
        
    if os.path.exists(S_graph_path):
        logging.info(f"Output file {S_graph_path} already exists, skipping entire processing pipeline")
        # Load the existing graph to return
        logging.info(f"Loading existing graph from {S_graph_path}")
        graph, _ = dgl.load_graphs(S_graph_path)
        return graph[0]
    
    logging.info("Initializing data loader...")
    dataloader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, genre_path, device)
    
    logging.info("Generating EM-based strong interest graph...")
    denoised_graph = dataloader.Get_S_views(dataloader.graph, save_dir)
    
    logging.info("EM completed successfully!")
    logging.info(f"Original graph: {dataloader.graph.number_of_edges('play')} play edges")
    logging.info(f"Denoised graph: {denoised_graph.number_of_edges('play')} play edges")
    logging.info(f"Removed {dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play')} play edges ({(dataloader.graph.number_of_edges('play') - denoised_graph.number_of_edges('play'))/dataloader.graph.number_of_edges('play')*100:.2f}%)")
    
    return denoised_graph

if __name__ == "__main__":
    denoised_graph = main()
    print("EM-based strong interest graph generation completed!")

# graph_p1
# INFO:root:EM completed successfully!
# INFO:root:Original graph: 3818315 play edges
# INFO:root:Denoised graph: 1693550 play edges
# INFO:root:Removed 2124765 play edges (55.65%)
# EM-based strong interest graph generation completed!

# graph_p2(right)
# INFO:root:EM completed successfully!
# INFO:root:Original graph: 3818315 play edges
# INFO:root:Denoised graph: 2132555 play edges
# INFO:root:Removed 1685760 play edges (44.15%)
# EM-based strong interest graph generation completed!

# graph_p2(left)
# INFO:root:EM completed successfully!
# INFO:root:Original graph: 3818315 play edges
# INFO:root:Denoised graph: 1358215 play edges
# INFO:root:Removed 2460100 play edges (64.43%)
# EM-based strong interest graph generation completed!

# graph_0.5_0.5
# INFO:root:EM completed successfully!
# INFO:root:Original graph: 3818315 play edges
# INFO:root:Denoised graph: 1342114 play edges
# INFO:root:Removed 2476201 play edges (64.85%)
# EM-based strong interest graph generation completed!

# graph_0.7_0.3
# INFO:root:EM completed successfully!
# INFO:root:Original graph: 3818315 play edges
# INFO:root:Denoised graph: 1377473 play edges
# INFO:root:Removed 2440842 play edges (63.92%)
# EM-based strong interest graph generation completed!