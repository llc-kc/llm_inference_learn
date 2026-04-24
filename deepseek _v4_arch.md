# DeepSeek-V4-Pro 架构详解：MCH（Manifold-Constrained Hyper-Connections）与混合注意力机制

## 整体架构概览

DeepSeek-V4-Pro 是一个 **Mixture-of-Experts (MoE)** 模型，总参数量 **1.6T**，每 token 激活 **49B** 参数，支持 **1M token** 上下文长度。代码结构如下：

```plain
text复制Transformer
├── ParallelEmbedding          (词汇嵌入，沿词表维度并行切分)
│
├── [Block × 61层]
│   ├── Attention (MLA)        (Multi-head Latent Attention + 混合稀疏注意力)
│   │   ├── MLA核心 (低秩Q投影 + KV计算)
│   │   ├── Compressor         (KV缓存压缩)
│   │   └── Indexer (可选)     (压缩KV位置选择器)
│   │
│   └── MoE (FFN)              (混合专家)
│       ├── Gate               (路由门控)
│       ├── [Expert × 384]     (SwiGLU FFN, FP4量化)
│       └── Shared_Expert      (共享专家)
│
├── RMSNorm
├── ParallelHead               (输出投影，HC加权合并 → 词表logits)
│
└── [MTPBlock × 1层]            (Multi-Token Prediction)
```

配置文件 (`inference/config.json`) 的关键参数：

| 参数                  | 值   | 说明                            |
| :-------------------- | :--- | :------------------------------ |
| `dim`                 | 7168 | 隐藏层维度                      |
| `n_layers`            | 61   | Transformer层数                 |
| `n_heads`             | 128  | 注意力头数                      |
| `head_dim`            | 512  | 每个注意力头的维度              |
| `rope_head_dim`       | 64   | RoPE应用的子维度                |
| `n_routed_experts`    | 384  | 路由专家总数                    |
| `n_activated_experts` | 6    | 每token激活的专家数             |
| `n_shared_experts`    | 1    | 共享专家数                      |
| `q_lora_rank`         | 1536 | Q的低秩投影维度                 |
| `o_lora_rank`         | 1024 | O的低秩投影维度                 |
| `o_groups`            | 16   | 输出分组的数量                  |
| `window_size`         | 128  | 局部滑动窗口大小                |
| `hc_mult`             | 4    | HC (Hyper-Connections) 乘法因子 |
| `hc_sinkhorn_iters`   | 20   | Sinkhorn迭代次数                |

------

## 1. MCH — Manifold-Constrained Hyper-Connections

### 核心理念

传统的Residual Connection是将每层的输出直接加到输入上：

```plain
text复制x_out = x_in + F(x_in)
```

MCH将这种**单标量残差**扩展为一个**流形上的线性混合**，在 `hc_mult` 个并行副本之间做**线性组合**。当 `hc_mult=4` 时，每个位置维护 **4个并行状态副本**，它们之间通过一个可学习的组合矩阵进行混合。

### 代码实现

#### 1.1 状态初始化与传播 (`Transformer.forward`, line 806-808)

```python
python复制# 从 embed 后，将单个状态扩展到 hc_mult 个副本
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, S, hc_mult, D]
# 每一层都返回 [B, S, hc_mult, D]
for layer in self.layers:
    h = layer(h, start_pos, input_ids)
# 最后用 HC head 合并回 [B, S, D]
logits = self.head(h, self.hc_head_fn, ...)
```

#### 1.2 每一层的HC处理 (`Block.forward`, line 689-701)

每层包含两个HC阶段：**注意力前** 和 **FFN前**：

```python
python复制def forward(self, x, start_pos, input_ids):
    residual = x                               # [B, S, hc_mult, D] 保存所有副本
    
    # ===== Attention 前的 HC 前处理 =====
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.attn_norm(x)                      # [B, S, D] 单向量
    x = self.attn(x, start_pos)                # 注意力计算
    x = self.hc_post(x, residual, post, comb)  # [B, S, hc_mult, D] 恢复多副本
    
    # ===== FFN 前的 HC 前处理 =====
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, ...)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x
```

#### 1.3 HC 前处理 — 从多副本缩减到单向量 (`hc_pre`, line 674-682)

这是最关键的部分。`hc_mult`个副本通过一个可学习的 **mixing matrix** 进行加权组合：

```plain
text复制输入形状: x [B, S, hc, D]
         hc_fn [mix_hc, hc*D]    其中 mix_hc = (2 + hc) * hc = 24
         hc_scale [3]             三个通道的缩放因子
         hc_base [mix_hc]         偏置

步骤:
1. 将 x 展平: x [B, S, hc*D]
2. 乘以 hc_fn 生成混合系数: mixes [B, S, mix_hc]
3. 应用 RMS 归一化（控制系数范数）
4. 用 Sinkhorn 算子将 mixes 分解为三部分:
   - pre[hc]  — 前向权重 (sigmoid 激活)
   - post[hc] — 后向权重 (2 × sigmoid 激活)
   - comb[hc, hc] — 副本间组合矩阵 (Sinkhorn 双随机化)
5. y = Σ(pre[i] * x[:,:,i,:]) — 加权求和得到单一输出 [B, S, D]
```

**Sinkhorn 算子** (`hc_split_sinkhorn_kernel`, [kernel.py](http://kernel.py/) line 372-428)：

- 将24维的mixes拆分为三块：pre(4维) + post(4维) + comb(16维)
- pre用 `sigmoid + eps` 保证正数
- post用 `2 * sigmoid`
- comb用 `softmax(-1) + eps` → 迭代Sinkhorn归一化（默认20次）得到**双重随机矩阵**（每行每列之和均为1的正矩阵）
- 双重随机的 comb 矩阵确保信息在各副本间**均匀地双向流动**

#### 1.4 HC 后处理 — 从单向量扩展回多副本 (`hc_post`, line 684-687)

```plain
text复制输出形状: x [B, S, D]          (注意力/FFN的结果)
         residual [B, S, hc, D]  (原始多副本残差)
         post [B, S, hc]         (后向权重)
         comb [B, S, hc, hc]     (组合矩阵，双重随机)

计算:
  y = post[i] * x + Σ(comb[i,j] * residual[j])
  y 形状: [B, S, hc, D]
```

这意味着：

- **post** 决定了新信息向每个副本注入的强度
- **comb** 是一个双重随机矩阵，决定各副本之间旧信息的混合比例
- 新信息通过注意力/FFN的计算得到，旧信息来自残差的多副本状态

#### 1.5 HC Head — 最终合并输出 (`ParallelHead.hc_head`, line 729-736)

在最后一层之后，需要将多副本合并回单向量来预测logits：

```python
python复制def hc_head(self, x, hc_fn, hc_scale, hc_base):
    # x: [B, S, hc, D]
    x = x.flatten(2).float()            # [B, S, hc*D]
    mixes = F.linear(x, hc_fn)          # [B, S, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    y = Σ(pre[i] * x[:,:,i,:])          # 加权求和
    return y
```

这里只使用 pre 权重（sigmoid激活 + eps）做加权求和，没有 comb 矩阵。

### MCH 的数学本质

将HC过程抽象为：

**前处理（降维）：**

1. 多副本状态 `X ∈ ℝ^{hc×D}` 通过线性映射得到 `mixes ∈ ℝ^{(2+hc)·hc}`
2. 分解为 `pre, post, comb`，其中 `comb` 经Sinkhorn正则化成为**双重随机矩阵**
3. 缩减输出：`x̄ = Σ preᵢ · Xᵢ`

**后处理（升维）：**

```plain
text复制Yᵢ = postᵢ · x̄ + Σ combᵢⱼ · residualⱼ
    ↑新信息        ↑旧信息的组合混合
```

**Manifold-Constrained** 的含义：

- `comb` 矩阵被约束为**双重随机矩阵**（行和=列和=1），这等价于一个**马尔可夫链转移矩阵**，确保副本间的信息流保持概率守恒
- `pre` 和 `post` 权重使用 sigmoid 激活，确保值在 `(0,1)` 范围内
- 整体上看，HC将残差路径从**单标量加性跳跃**扩展为**hc维流形上的几何变换**，每个副本代表流形上的一个方向

与标准残差的对比：

| 特性       | 标准残差   | MCH (hc_mult=4)                |
| :--------- | :--------- | :----------------------------- |
| 副本数     | 1          | 4                              |
| 信息流     | 单一直连   | 4维流形上的线性混合            |
| 混合矩阵   | 无         | 双重随机矩阵（Sinkhorn正则化） |
| 可学习参数 | 0          | `(2+hc)*hc*(hc*D + 1)` 个      |
| 信号传播   | 简单的加法 | 流形约束下的加权组合           |
| 状态多样性 | 无         | 4个并行表示，通过正则化混合    |

------

## 2. 混合注意力架构 — CSA + HCA

DeepSeek-V4的注意力机制分为**三个并行组件**：

### 2.1 局部滑动窗口注意力 (Sliding Window Attention)

每层固定维护一个大小为 `window_size=128` 的环形KV缓存：

```python
python复制# 从输入到KV
kv = self.wkv(x)                    # [B, S, D] → [B, S, head_dim]
kv = self.kv_norm(kv)               # RMS归一化
apply_rotary_emb(kv[..., -rd:], ...)# 对后64维应用RoPE
act_quant(kv[..., :-rd], ...)       # 非RoPE部分做FP8模拟量化

# 缓存管理（环形缓冲区）
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)  # 覆盖最旧位置
```

每个位置只能关注它前面最多128个token，这为大上下文提供了**亚线性计算**的基础。

### 2.2 压缩稀疏注意力 (CSA — Compressed Sparse Attention)

对于超出滑动窗口的历史，DeepSeek-V4不保留全精度KV，而是通过 **Compressor** 将KV缓存压缩后再存储。

#### Compressor (`Compressor`类，line 279-377)

核心思想：对连续的 `compress_ratio` 个token做**门控池化压缩**，输出一个压缩后的KV向量。

```plain
text复制输入: K,V 序列 [长度 = L]
      compress_ratio = R

过程:
1. wkv(x) → kv                    # 线性投影到 head_dim
2. wgate(x) → score               # 门控分数
3. kv = kv.unflatten(1, (-1, R))  # 分组: [B, L/R, R, D]
4. score = softmax(score + ape)   # 带可学习位置编码的softmax权重
5. kv = Σ(score * kv, dim=2)      # 注意力池化 → [B, L/R, D]
6. Norm + RoPE                    # 归一化后加位置编码
7. 存入kv_cache[win:]
```

关键细节：

- **重叠压缩** (`overlap=True`, `ratio=4`)：当压缩比为4时，使用重叠窗口，相邻压缩块共享边界token，减少块边界信息损失
- **掩码机制**：在prefill阶段使用因果掩码，保证压缩时只看历史token
- **增量压缩**：decode阶段逐步累加，每 `compress_ratio` 步触发一次压缩

**压缩比配置** (`compress_ratios`)：61层各有不同压缩比，大部分层是 `4` 或 `128`，最后一层为 `0`（不压缩，纯滑动窗口）。

#### Indexer (`Indexer`类，line 380-433)

对于 `compress_ratio=4` 的层，压缩后的KV仍然太多（1M token → 250K压缩位置），不能全部参与注意力。Indexer学习一个**筛选函数**，从中选出 **top-k** 个压缩KV位置：

```python
python复制# 独立的低秩Q投影
q = self.wq_b(qr)                  # low-rank query → 64 heads × 128 dim
q = rotate_activation(q)           # Hadamard旋转（改善量化分布）
fp4_act_quant(q)                   # FP4量化模拟

# 压缩KV (通过内部的Compressor)
self.compressor(x, start_pos)

# 评分: q 与所有压缩KV做点积
index_score = einsum("bshd,btd->bsht", q, kv_cache)
index_score = (index_score.relu() * weights).sum(dim=2)  # 加权合并多头的分数

# 选择topk位置
topk_idxs = index_score.topk(index_topk=1024)
```

Indexer使用**独立的低维空间**（64 head × 128 dim，对比MLA的128 head × 512 dim），并经过：

1. Hadamard旋转 → 2. FP4量化模拟 → 3. 注意力评分 → 4. top-k选择

### 2.3 压缩比128的重度压缩注意力 (HCA — Heavily Compressed Attention)

对于 `compress_ratio=128` 的层，1M token被压缩为约8192个KV位置，此时**不借助Indexer**，直接用公式化的索引选择：

```python
python复制compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
```

这相当于对压缩后的KV做**全注意力**（因为8192个位置对所有注意力头来说都已足够小）。

### 2.4 三种注意力组件的整合

所有61层的压缩比配置（从config.json解析）：

```plain
text复制[128, 128, 4, 128, 4, 128, 4, 128, ... , 4, 0]
     ↑         ↑         ↑              ↑    ↑
    第0层     第2层     第4层        第59层  第60层
```

具体合并时：

```python
python复制# 1. 窗口内KV（当前128个token）—— 全精度
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)

# 2. 压缩KV索引
if self.indexer is not None:         # ratio=4 → 学习式筛选
    compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
else:                                # ratio=128 → 全选
    compress_topk_idxs = get_compress_topk_idxs(ratio, ...)

# 3. 合并
topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

# 4. 稀疏注意力计算（GPU kernel）
o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
```

`sparse_attn` kernel ([kernel.py](http://kernel.py/) line 277-368) 实现了一种**FlashAttention风格的在线softmax**：

- 按 `block_size=64` 分块
- 每块根据 `topk_idxs` 从KV缓存中gather所需条目
- 使用 `attn_sink` 可学习偏置（每个头独立）
- 数值稳定的"running max/sum"模式

### 2.5 MLA — Multi-head Latent Attention

注意力的核心采用MLA架构（第436-543行）：

```python
python复制# Q: 低秩分解 (dim → q_lora_rank → n_heads * head_dim)
qr = self.q_norm(self.wq_a(x))       # 7168 → 1536 (低秩)
q = self.wq_b(qr)                    # 1536 → 128*512 = 65536
q *= rsqrt(q.square().mean(-1) + eps) # QK归一化

# KV: 单头 (n_kv_heads = 1)
kv = self.wkv(x)                     # 7168 → 512
kv = self.kv_norm(kv)                # 对KV也做归一化

# RoPE: 只对后64维应用
apply_rotary_emb(q[..., -rd:], freqs_cis)
apply_rotary_emb(kv[..., -rd:], freqs_cis)

# O: 分组低秩投影
# o heads 先按 n_groups 分组归约
o = o.view(b, s, n_local_groups, -1)
# wo_a: group-level 低秩投影
wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
o = einsum("bsgd,grd->bsgr", o, wo_a)  # 分组归约
# wo_b: 恢复原始维度
x = self.wo_b(o.flatten(2))
```

MLA的关键特点是：

- **KV只有1个头**（MQA风格），大幅降低KV缓存
- **Q用低秩分解**（dim→q_lora_rank→n_heads*head_dim），中间有RMSNorm
- **QK归一化**：`q *= rsqrt(q.square().mean(-1))` 稳定注意力分数
- **O用分组低秩投影**：`n_groups=16`，每组先压缩再恢复
- **RoPE只应用于64维**，剩余448维是"无位置"的，可被有效压缩

------

## 3. MoE 专家混合

### 3.1 门控机制 (`Gate`类，line 546-584)

```python
python复制scores = linear(x, self.weight)      # 路由分数
if score_func == "sqrtsoftplus":
    scores = F.softplus(scores).sqrt()  # √softplus 激活
# bias只影响topk选择，不影响路由权重
scores_with_bias = scores + self.bias
indices = scores_with_bias.topk(6)     # top-6专家
weights = original_scores.gather(1, indices)  # 原始分数作为路由权重
weights /= weights.sum(dim=-1)         # 归一化
weights *= route_scale(2.5)           # 路由缩放
```

前3层 (`n_hash_layers=3`) 使用**哈希路由**——专家索引由token ID预先决定，不做学习：

```python
python复制if self.hash:
    indices = self.tid2eid[input_ids]  # 查表得到预分配的专家
```

### 3.2 专家计算 (`Expert`类，line 587-606)

每个专家是SwiGLU FFN，在FP4精度下存储和计算：

```python
python复制def forward(self, x, weights=None):
    gate = self.w1(x).float()    # Linear in FP4
    up = self.w3(x).float()      # Linear in FP4
    # SwiGLU: x = SiLU(gate) * up
    x = F.silu(gate) * up
    if weights is not None:
        x = weights * x
    return self.w2(x)            # Linear in FP4
```

`swiglu_limit=10.0` 对激活值做截断，防止量化溢出。

### 3.3 专家并行 (`MoE`类，line 609-645)

384个专家沿tensor parallelism维度切分，每个rank负责 `384 // world_size` 个专家。使用 `bincount` 统计每个专家被选中的次数，只激活非空专家：

```python
python复制counts = torch.bincount(indices.flatten(), minlength=n_routed_experts)
for i in range(experts_start_idx, experts_end_idx):
    if counts[i] == 0: continue
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top])
```

最后加上共享专家的输出：

```python
python复制y += self.shared_experts(x)  # 共享专家对所有token都激活
```

------

## 4. 量化策略

| 组件                  | 精度               | 说明                      |
| :-------------------- | :----------------- | :------------------------ |
| 注意力权重 (除wo_a外) | FP8 (e4m3)         | 每128个元素一组，动态量化 |
| wo_a权重              | FP8                | 分组量化                  |
| 专家权重              | FP4 (e2m1fn)       | 每32个元素一组，幂次缩放  |
| 其他权重 (embed等)    | BF16               | 全精度                    |
| 激活值 (推理过程)     | BF16 → FP8模拟量化 | 非RoPE维度做QAT模拟       |
| Indexer激活           | FP4模拟量化        | Hadamard旋转+FP4          |

**FP8 GEMM** (`fp8_gemm_kernel`, [kernel.py](http://kernel.py/) line 203-273)：

- A和B都是FP8，block_size=128
- 每个block_K=128由独立的scaling factor缩放
- 使用双累加器（C_local + C_local_accum）提高精度

**FP4 GEMM** (`fp4_gemm_kernel`, [kernel.py](http://kernel.py/) line 441-515)：

- A是FP8 (block_size=128)，B是FP4 (block_size=32)
- FP4先升为FP8（通过FP32中转），再做FP8×FP8 GEMM
- 激活scale和权重scale分别应用到累加器

------

## 5. MTP — Multi-Token Prediction

在最后一层之后有一个额外的 `MTPBlock`（第739-767行），用于同时预测下一个token和后续token：

```python
python复制def forward(self, x, start_pos, input_ids):
    e = self.embed(input_ids)             # 输入token嵌入
    e = self.enorm(e)                     # 嵌入归一化
    x = self.hnorm(x)                     # 隐藏状态归一化
    x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)  # 嵌入信息注入
    x = super().forward(x, start_pos, input_ids)       # 标准Block处理
    logits = self.head(x, hc_head_fn, hc_head_scale, hc_head_base, self.norm)
    return logits
```

MTP通过任务特定的HC head（独立的`hc_head_fn`参数）预测logits。

------

## 6. RoPE 与 YaRN 扩展

DeepSeek-V4使用YaRN（Yet another RoPE extensioN）将上下文从65536扩展到100万：

```python
python复制# 频率插值
freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
if original_seq_len > 0:
    low, high = find_correction_range(beta_fast, beta_slow, ...)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
```

关键参数：`rope_factor=16`（频率缩放16倍），`original_seq_len=65536`。

压缩KV使用独立的RoPE基频率：`compress_rope_theta=160000`。

------

## 7. 效率总结

| 指标                | DeepSeek-V3.2 | DeepSeek-V4-Pro | 提升       |
| :------------------ | :------------ | :-------------- | :--------- |
| 1M token预填充FLOPs | 100%（基准）  | 27%             | **3.7×**   |
| KV缓存大小          | 100%（基准）  | 10%             | **10×**    |
| 总参数量            | 671B          | 1.6T            | 更大但稀疏 |
| 激活参数            | 37B           | 49B             | 更大       |

效率提升主要来自：

1. **CSA**：压缩比128/4大幅减少KV条目
2. **Indexer**：只保留top-1024个压缩KV位置做注意力
3. **窗口注意力**：window_size=128的局部注意力是O(n)而不是O(n²)
4. **FP4专家**：权重存储减半，GEMM更快
5. **MLA**：单头KV + 低秩投影减少计算和存储

这套混合注意力设计使DeepSeek-V4-Pro在处理100万token时，只需要DeepSeek-V3.2 **27%的计算量**和**10%的KV缓存**。