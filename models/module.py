import dgl
import torch
import os
from scipy.sparse import csr_matrix

import torch
import dgl
import math

def build_user_user_graph(graph_S, save_path=None, min_common=15, block_size=2000):
    """
    分块在 GPU 上根据强兴趣用户-游戏图 graph_S 构建用户-用户图。
    支持大规模用户图（6w+），显存安全。

    参数：
        graph_S: dgl.DGLHeteroGraph
        save_path: str 或 None
        min_common: int
        block_size: int, 每次处理多少个用户块（显存控制关键参数）
    返回：
        graph_UU: dgl.DGLGraph
    """
    device = graph_S.device
    u, g = graph_S.edges(etype=('user', 'play', 'game'))
    num_users = graph_S.num_nodes('user')
    num_games = graph_S.num_nodes('game')

    # 构建稀疏邻接矩阵 (U × G)
    values = torch.ones(len(u), device=device)
    adj = torch.sparse_coo_tensor(
        torch.stack([u, g]), values, (num_users, num_games)
    ).coalesce()

    src_all, dst_all = [], []

    # 分块计算
    num_blocks = math.ceil(num_users / block_size)
    print(f"⚙️ 用户共 {num_users}，分为 {num_blocks} 个块，每块 {block_size} 用户")

    for i in range(num_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, num_users)

        # 提取当前块的行
        idx_i = torch.arange(start_i, end_i, device=device)
        adj_i = adj.index_select(0, idx_i)

        # 计算当前块与所有用户的交集
        common = torch.sparse.mm(adj_i, adj.transpose(0, 1)).to_dense()

        # 找到满足 min_common 的边
        mask = (common >= min_common)
        src, dst = mask.nonzero(as_tuple=True)

        # 映射回全局用户编号
        src = src + start_i
        dst = dst

        # 去除自环
        mask2 = src != dst
        src, dst = src[mask2], dst[mask2]

        src_all.append(src.cpu())
        dst_all.append(dst.cpu())

        torch.cuda.empty_cache()
        print(f"✅ 已处理块 {i+1}/{num_blocks} ({end_i-start_i} 用户)，累计边数 {sum(len(s) for s in src_all)}")

    # 拼接所有结果
    src_all = torch.cat(src_all)
    dst_all = torch.cat(dst_all)

    # 构建用户–用户图
    graph_UU = dgl.graph((src_all, dst_all), num_nodes=num_users)
    print(f"🎯 用户–用户图构建完成：{graph_UU.num_nodes()} 个节点, {graph_UU.num_edges()} 条边")

    if save_path is not None:
        dgl.save_graphs(save_path, [graph_UU])
        print(f"💾 用户–用户图已保存到: {save_path}")

    return graph_UU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

# ====== 局部注意力模块 ======
class LocalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=64, mode='sa'):
        super().__init__()
        self.mode = mode
        self.dim = dim 
        self.num_heads = num_heads
        self.d_k = self.dim // num_heads
        self.window_size = window_size

        # 分别处理两路输入
        self.W_Q = nn.Linear(dim, self.dim)
        self.W_K = nn.Linear(dim, self.dim)
        self.W_V = nn.Linear(dim, self.dim)

        self.W_Q_s = nn.Linear(dim, self.dim)
        self.W_K_s = nn.Linear(dim, self.dim)
        self.W_V_s = nn.Linear(dim, self.dim)

        self.out_proj = nn.Linear(self.dim, dim)
        self.out_proj_s = nn.Linear(self.dim, dim)

        self.dropout = Dropout(0.1)

        if self.mode == 'ma':
            self.w11 = nn.Parameter(torch.tensor(0.5))
            self.w12 = nn.Parameter(torch.tensor(0.5))
            self.w21 = nn.Parameter(torch.tensor(0.5))
            self.w22 = nn.Parameter(torch.tensor(0.5))

    def _window_partition(self, x):
        """将序列按窗口切块 (L, D) → (num_windows, window_size, D)"""
        L, D = x.shape
        pad_len = (self.window_size - (L % self.window_size)) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        L_pad = x.shape[0]
        num_windows = L_pad // self.window_size
        x = x.view(num_windows, self.window_size, D)
        return x, L_pad, pad_len

    def _window_reverse(self, x, L_ori, pad_len):
        """将窗口拼回原序列形状"""
        x = x.view(-1, x.shape[-1])
        if pad_len > 0:
            x = x[:-pad_len]
        return x

    def _local_attention(self, Q, K, V):
        """在每个窗口内部计算注意力"""
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out

    def forward(self, h, h1):
        # h, h1: (L, D)
        L, D = h.shape
        window_size = self.window_size

        # ====== Step 1. 窗口划分 ======
        h_win, L_pad, pad_len = self._window_partition(h)
        h1_win, _, _ = self._window_partition(h1)

        num_windows = h_win.shape[0]

        # ====== Step 2. 线性变换 ======
        def proj(W_Q, W_K, W_V, x):
            Q = W_Q(x)
            K = W_K(x)
            V = W_V(x)
            return Q, K, V

        Q, K, V = proj(self.W_Q, self.W_K, self.W_V, h_win)
        Qs, Ks, Vs = proj(self.W_Q_s, self.W_K_s, self.W_V_s, h1_win)

        # ====== Step 3. 拆分多头 (num_windows, win, D) → (num_windows, H, win, d_k)
        def split_heads(x):
            return x.view(num_windows, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)
        Qs = split_heads(Qs)
        Ks = split_heads(Ks)
        Vs = split_heads(Vs)

        # ====== Step 4. 局部注意力计算 ======
        # out_local = self._local_attention(Q, K, V)
        # out_local_s = self._local_attention(Qs, Ks, Vs)

        # # ====== Step 5. 互注意力 (mode='ma') ======
        # if self.mode == 'ma':
        out_cross = self._local_attention(Q, Ks, Vs)
        out_cross_s = self._local_attention(Qs, K, V)

            # out_local = self.w11 * out_local + self.w12 * out_cross
            # out_local_s = self.w22 * out_local_s + self.w21 * out_cross_s
        out_local = out_cross
        out_local_s = out_cross_s

        # ====== Step 6. 合并多头 & 输出 ======
        def merge_heads(x):
            x = x.transpose(1, 2).contiguous().view(num_windows * window_size, self.dim)
            return x

        out = merge_heads(out_local)
        out_s = merge_heads(out_local_s)

        out = self._window_reverse(out, L, pad_len)
        out_s = self._window_reverse(out_s, L, pad_len)

        out = self.out_proj(out)
        out_s = self.out_proj_s(out_s)

        return out, out_s


# ====== MLP 模块 ======
class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act_fn = nn.ReLU(inplace=True)
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ====== Attention Block 模块 ======
class Attention_Block(nn.Module):
    def __init__(self, dim, mode='sa', window_size=64):
        super().__init__()
        self.dim = dim
        self.attention_norm_x = nn.LayerNorm(dim, eps=1e-5)
        self.attention_norm_y = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm_x = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm_y = nn.LayerNorm(dim, eps=1e-5)

        self.attn = LocalAttention(dim, num_heads=4, window_size=window_size, mode=mode)
        self.ffn_x = Mlp(dim)
        self.ffn_y = Mlp(dim)

    def forward(self, x, y):
        # --- Attention + 残差 ---
        residual_x = x
        residual_y = y
        x_norm = self.attention_norm_x(x)
        y_norm = self.attention_norm_y(y)

        x_out, y_out = self.attn(x_norm, y_norm)
        x = x_out + residual_x
        y = y_out + residual_y

        # # --- FeedForward + 残差 ---
        # residual_x = x
        # residual_y = y
        # x_norm = self.ffn_norm_x(x)
        # y_norm = self.ffn_norm_y(y)
        # x = self.ffn_x(x_norm) + residual_x
        # y = self.ffn_y(y_norm) + residual_y

        return x, y
    
class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_list):
        """
        h_list: List[Tensor], 每个张量形状为 [N, hidden_dim]
        表示同一批节点的多源嵌入（例如来自不同图视角）
        """
        # [num_views, N, hidden_dim]
        h_stack = torch.stack(h_list, dim=0)

        # 计算注意力分数 [num_views, N, 1]
        score = self.attn(torch.tanh(self.proj(h_stack)))

        # 沿不同来源方向做 softmax
        attn_weights = F.softmax(score, dim=0)  # [num_views, N, 1]

        # 加权求和
        h_fused = torch.sum(attn_weights * h_stack, dim=0)

        return h_fused, attn_weights   

def block_similarity_fusion(self,h_user_s, h_user_f, block_size=2048, device=None):
        """
        分块计算 (h_user_s_norm @ h_user_s_norm.T)，防止显存爆炸。
        计算结果与原逻辑：
            h_user_s_norm = F.normalize(h_user_s, dim=1)
            h_h = (h_user_s_norm @ h_user_s_norm.T) / sqrt(dim)
            h_h = F.softmax(h_h, dim=1)
            h_user_f = h_h @ h_user_f + h_user_f
        等价。

        参数：
            h_user_s: torch.Tensor, [N, D]
            h_user_f: torch.Tensor, [N, D]
            block_size: 每次处理的块大小
            device: 运算设备（默认同 h_user_s）

        返回：
            h_user_f_new: torch.Tensor, [N, D]
        """
        if device is None:
            device = h_user_s.device

        N, D = h_user_s.size()
        h_user_s_norm = F.normalize(h_user_s, dim=1)

        h_user_f_new = torch.zeros_like(h_user_f, device=device)

        sqrt_d = math.sqrt(D)

        # 分块计算 h_h @ h_user_f
        for start_i in range(0, N, block_size):
            end_i = min(start_i + block_size, N)
            h_i = h_user_s_norm[start_i:end_i]  # [b, D]

            # 分块计算 softmax 权重 (按行)
            scores_row = []
            for start_j in range(0, N, block_size):
                end_j = min(start_j + block_size, N)
                h_j = h_user_s_norm[start_j:end_j]  # [b2, D]

                # 计算块间相似度
                sim_block = (h_i @ h_j.T) / sqrt_d  # [b, b2]
                scores_row.append(sim_block)

            # 拼接整行的相似度并 softmax
            sim_row = torch.cat(scores_row, dim=1)
            sim_row = F.softmax(sim_row, dim=1)

            # 再次分块乘 h_user_f
            out_row_parts = []
            for start_j in range(0, N, block_size):
                end_j = min(start_j + block_size, N)
                out_row_parts.append(sim_row[:, start_j:end_j] @ h_user_f[start_j:end_j])
            out_row = torch.stack(out_row_parts, dim=0).sum(dim=0)  # [b, D]

            h_user_f_new[start_i:end_i] = out_row + h_user_f[start_i:end_i]

            # 释放中间显存
            del h_i, sim_row, scores_row, out_row_parts, out_row
            torch.cuda.empty_cache()

        return h_user_f_new
    
def block_topk_similarity_fusion(self,h_user_s, h_user_f, k=10, block_size=2048, device=None):
    """
    分块计算基于 top-k 的相似度注意力加权更新：
        1. 对 h_user_s 做归一化；
        2. 分块计算相似度矩阵 (h_user_s_norm @ h_user_s_norm.T)；
        3. 每行取 top-k；
        4. 对 top-k 相似度做归一化；
        5. 用加权求和更新 h_user_f。

    参数：
        h_user_s: torch.Tensor, [N, D]
        h_user_f: torch.Tensor, [N, D]
        k: int，保留的 top-k 相似邻居数
        block_size: int，分块大小
        device: 运算设备（默认同 h_user_s）

    返回：
        h_user_f_new: torch.Tensor, [N, D]
    """
    if device is None:
        device = h_user_s.device

    N, D = h_user_s.size()
    h_user_s_norm = F.normalize(h_user_s, dim=1)
    h_user_f_new = torch.zeros_like(h_user_f, device=device)
    sqrt_d = math.sqrt(D)

    for start_i in range(0, N, block_size):
        end_i = min(start_i + block_size, N)
        h_i = h_user_s_norm[start_i:end_i]  # [b, D]

        # 存储当前块的相似度结果
        all_scores = []

        # 分块计算相似度
        for start_j in range(0, N, block_size):
            end_j = min(start_j + block_size, N)
            h_j = h_user_s_norm[start_j:end_j]  # [b2, D]

            sim_block = (h_i @ h_j.T) / sqrt_d  # [b, b2]
            all_scores.append(sim_block)

            del h_j, sim_block
            torch.cuda.empty_cache()

        # 拼接整行相似度
        sim_row = torch.cat(all_scores, dim=1)  # [b, N]
        del all_scores
        torch.cuda.empty_cache()

        # 取 top-k
        values, indices = torch.topk(sim_row, k=k, dim=1)

        # 构造稀疏 mask 矩阵
        mask = torch.zeros_like(sim_row)
        mask.scatter_(1, indices, 1.0)

        # 保留 top-k 相似度
        sim_row = sim_row * mask

        # 行归一化
        sim_row = F.normalize(sim_row, p=1, dim=1)

        # 用相似度加权更新 h_user_f
        out_row = torch.zeros((end_i - start_i, D), device=device)
        for start_j in range(0, N, block_size):
            end_j = min(start_j + block_size, N)
            out_row += sim_row[:, start_j:end_j] @ h_user_f[start_j:end_j]

        h_user_f_new[start_i:end_i] = out_row + h_user_f[start_i:end_i]

        del h_i, sim_row, mask, values, indices, out_row
        torch.cuda.empty_cache()

    return h_user_f_new