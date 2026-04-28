import torch
import dgl
import os
from parser import parse_args

def normalize_by_user_node(graph, edge_weights, edge_type):

    src, dst = graph.edges(etype=edge_type)
    
    src_to_indices = {}
    for i, s in enumerate(src):
        s_item = s.item()
        if s_item not in src_to_indices:
            src_to_indices[s_item] = []
        src_to_indices[s_item].append(i)
    
    normalized_weights = torch.zeros_like(edge_weights)
    for s_item, indices in src_to_indices.items():
        curr_weights = edge_weights[indices]
        weight_sum = curr_weights.sum()
        if weight_sum > 0:
            normalized_weights[indices] = curr_weights / weight_sum
    
    return normalized_weights

args = parse_args()
graph_path = "data_exist/mrw.bin"
weights_path = "data_exist/weights_mrw.pth"
output_path = "data_exist/weights_mrw_normalized.pth"
edge_type = "plays"  

print(f"Loading graph file: {graph_path}")
graph, _ = dgl.load_graphs(graph_path)
graph = graph[0]  

print(f"Loading weight file: {weights_path}")
edge_weights = torch.load(weights_path)

print("\nOriginal weight statistics:")
print(f"Shape: {edge_weights.shape}")
print(f"Minimum: {edge_weights.min().item():.6f}")
print(f"Maximum: {edge_weights.max().item():.6f}")
print(f"Mean: {edge_weights.mean().item():.6f}")
print(f"Standard deviation: {edge_weights.std().item():.6f}")

num_edges = graph.num_edges(edge_type)
if num_edges != edge_weights.size(0):
    raise ValueError(f"Number of edges ({num_edges}) does not match the number of weights ({edge_weights.size(0)})!")

print(f"\nPerforming user-based normalization for edge type '{edge_type}'...")
normalized_weights = normalize_by_user_node(graph, edge_weights, edge_type)

print("\nNormalized weight statistics:")
print(f"Shape: {normalized_weights.shape}")
print(f"Minimum: {normalized_weights.min().item():.6f}")
print(f"Maximum: {normalized_weights.max().item():.6f}")
print(f"Mean: {normalized_weights.mean().item():.6f}")
print(f"Standard deviation: {normalized_weights.std().item():.6f}")

src, _ = graph.edges(etype=edge_type)
unique_src = torch.unique(src)

num_samples = min(5, len(unique_src))
sampled_src = unique_src[torch.randperm(len(unique_src))[:num_samples]]

print(f"\nValidating normalization results (randomly sampling {num_samples} users):")
for s in sampled_src:
    s_item = s.item()
    edge_indices = (src == s).nonzero().squeeze()
    
    orig_w = edge_weights[edge_indices]
    norm_w = normalized_weights[edge_indices]
    
    print(f"User ID: {s_item}")
    print(f"  Number of connected games: {len(edge_indices)}")
    print(f"  Sum of original weights: {orig_w.sum().item():.6f}")
    print(f"  Sum of normalized weights: {norm_w.sum().item():.6f}")
    print(f"  Original weight samples: {orig_w[:3].tolist() if len(orig_w)>=3 else orig_w.tolist()}")
    print(f"  Normalized weight samples: {norm_w[:3].tolist() if len(norm_w)>=3 else norm_w.tolist()}")
    print()

print(f"Saving normalized weights to: {output_path}")
torch.save(normalized_weights, output_path)
print("Done!")