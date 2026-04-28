import dgl
import torch
import numpy as np
from typing import Tuple
import scipy.sparse as sp

def build_adj_from_etype(g: dgl.DGLHeteroGraph,
                         etype: Tuple[str, str, str],
                         return_torch: bool = True):
    """
    从异构图中提取指定类型的边，构建邻接矩阵。

    参数：
        g: dgl.DGLHeteroGraph
        etype: 边类型，例如 ('user', 'play', 'game')
        return_torch: True 返回 torch.sparse_coo_tensor，False 返回 scipy.csr_matrix

    返回：
        邻接矩阵（稀疏），torch.sparse_coo_tensor 或 scipy.sparse.csr_matrix
    """
    # 获取边的两端索引（src=user节点, dst=game节点）
    src, dst = g.edges(etype=etype)  # DGL 在默认 backend 为 torch 时返回 torch.Tensor

    # 如果得到的是 numpy，转为 torch
    if not isinstance(src, torch.Tensor):
        src = torch.from_numpy(src)
    if not isinstance(dst, torch.Tensor):
        dst = torch.from_numpy(dst)

    # 确保为 long 类型（索引必须为整型）
    src = src.long()
    dst = dst.long()

    num_src = g.num_nodes(etype[0])   # user 节点数
    num_dst = g.num_nodes(etype[2])   # game 节点数

    if return_torch:
        values = torch.ones(src.shape[0], dtype=torch.float32, device=src.device)
        indices = torch.stack([src, dst], dim=0)  # shape (2, E)
        adj = torch.sparse_coo_tensor(indices, values, (num_src, num_dst))
        return adj.coalesce()
    else:
        # 转为 numpy 以构建 scipy 稀疏矩阵
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        data = (np.ones(src_np.shape[0], dtype=np.float32))
        adj = sp.csr_matrix((data, (src_np, dst_np)), shape=(num_src, num_dst))
        return adj
    
# def normalize_graph_mat(adj_mat):
#     '''
#     对称归一化邻接矩阵
#     '''
#     shape = adj_mat.get_shape()
#     rowsum = np.array(adj_mat.sum(1))
#     if shape[0] == shape[1]:
#         d_inv = np.power(rowsum, -0.5).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat_inv = sp.diags(d_inv)
#         norm_adj_tmp = d_mat_inv.dot(adj_mat)
#         norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
#     else:
#         d_inv = np.power(rowsum, -1).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat_inv = sp.diags(d_inv)
#         norm_adj_mat = d_mat_inv.dot(adj_mat)
#     return norm_adj_mat

def _normalize_torch_sparse(adj: torch.Tensor):
    """
    adj: torch.sparse_coo_tensor
    """
    assert adj.is_sparse

    adj = adj.coalesce()
    indices = adj.indices()   # (2, E)
    values = adj.values()     # (E,)
    device = values.device

    N, M = adj.shape

    # 行和（degree）
    row, col = indices
    deg = torch.zeros(N, device=device)
    deg.index_add_(0, row, values)

    if N == M:
        # ===== 对称归一化 D^{-1/2} A D^{-1/2} =====
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm_values = (
            deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
        )
    else:
        # ===== 左归一化 D^{-1} A =====
        deg_inv = torch.pow(deg, -1.0)
        deg_inv[torch.isinf(deg_inv)] = 0.0

        norm_values = deg_inv[row] * values

    return torch.sparse_coo_tensor(
        indices,
        norm_values,
        adj.shape,
        device=device
    ).coalesce()
    
def _normalize_scipy_sparse(adj_mat):
    shape = adj_mat.shape
    rowsum = np.array(adj_mat.sum(1)).flatten()

    if shape[0] == shape[1]:
        # 对称归一化
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
    else:
        # 左归一化
        d_inv = np.power(rowsum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)

    return norm_adj

def normalize_graph_mat(adj_mat):
    """
    对邻接矩阵进行归一化
    - scipy.sparse.csr_matrix
    - torch.sparse_coo_tensor

    方阵：D^{-1/2} A D^{-1/2}
    非方阵：D^{-1} A
    """

    # =========================
    # case 1: torch sparse
    # =========================
    if isinstance(adj_mat, torch.Tensor) and adj_mat.is_sparse:
        return _normalize_torch_sparse(adj_mat)

    # =========================
    # case 2: scipy sparse
    # =========================
    elif sp.issparse(adj_mat):
        return _normalize_scipy_sparse(adj_mat)

    else:
        raise TypeError(
            f"Unsupported matrix type: {type(adj_mat)}"
        )