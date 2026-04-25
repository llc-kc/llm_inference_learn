# DeepSeek V4 注意力机制详解：从原理到代码实现

DeepSeek V4 引入了 **Hybrid Attention Architecture**，融合了 **Compressed Sparse Attention (CSA)**、**Heavily Compressed Attention (HCA)**、**Sliding Window Attention (SWA)** 和 **DeepSeek Sparse Attention (DSA)**。下面我们从配置文件、核心类到底层 kernel，逐层深入。

------

## 一、总览：模型结构与压缩策略

### 1.1 config.json 中的关键配置

```plain
compress_ratios: [128, 128, 4, 128, 4, 128, 4, ... , 4, 0]
num_hidden_layers: 61
sliding_window: 128
index_topk: 1024
head_dim: 512, rope_head_dim: 64
q_lora_rank: 1536, o_lora_rank: 1024
compress_rope_theta: 160000
```

`compress_ratios` 数组长度为 62（61 层 + 最后一层为 0），每层的注意力压缩策略由其决定：

| compress_ratio 值 | 含义                                                  | 层数示例            |
| :---------------- | :---------------------------------------------------- | :------------------ |
| `0`               | 纯 Sliding Window Attention（无压缩）                 | 最后一层（索引 61） |
| `4`               | **c4a**：4x 压缩 + DSA（Indexer 做 top-k 稀疏注意力） | 第 2,4,6,… 层       |
| `128`             | **c128a**：128x 压缩 + 直接顺序索引（无 DSA）         | 第 0,1,3,5,… 层     |

两种压缩比率在整个 61 层中交替排列：`128, 128, 4, 128, 4, 128, 4, ...`，只有最后一层（最后一层 block 的注意力层）是 `0`，意味着它只做 SWA。

### 1.2 Attention 类初始化（model.py:436-482）

```python
class Attention(nn.Module):
    def __init__(self, layer_id, args):
        self.compress_ratio = args.compress_ratios[layer_id]
        self.window_size = args.window_size  # 128

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)  # DSA 的索引器
            else:
                self.indexer = None  # c128a 不跑 DSA

        # KV cache大小 = sliding window + 压缩后的KV
        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
```

**关键设计决策**：

- **c4a (ratio=4)**：配备 `Indexer`，它内部有一个 `Compressor` + top-k 选择，用于 DSA（稀疏注意力）
- **c128a (ratio=128)**：无 Indexer，直接用 `get_compress_topk_idxs` 获取**所有**压缩位置的索引（即稠密地 attend 全部压缩 token）

------

## 二、KV Cache 压缩核心：Compressor 类

### 2.1 基本原理

`Compressor` 是一个**学习到的门控池化**模块。它将连续的若干个 KV token（数量 = `compress_ratio`）通过可学习的加权求和，压缩成一个 token。

**数学形式**：

- 输入：连续 `r` 个 token 的 KV 投影值 `kv[t], kv[t+1], ..., kv[t+r-1]`
- 门控权重（gating score）：`score[t+i] = W_gate(x[t+i]) + position_ape[i]`
- 压缩后的单 token：`kv_compressed = softmax(score) · kv_pool`

其中 `x` 是压缩层的 hidden state 输入，`ape` 是可学习的绝对位置编码参数。

### 2.2 Prefill 阶段（start_pos == 0）源码分析（model.py:316-342）

```python
def forward(self, x, start_pos):
    kv = self.wkv(x)        # [b,s,d] -> [b,s,head_dim]
    score = self.wgate(x)   # 门控权重
    score += self.ape       # 加上可学习位置编码

    if start_pos == 0:      # Prefill 阶段
        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder

        # 保存尾部余数到状态缓冲区，用于后续 decode 增量
        if overlap and cutoff >= ratio:
            self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
            self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff]
        if remainder > 0:
            self.score_state[:bsz, offset:offset+remainder] = score[:, cutoff:]

        # 按 ratio 分组并 reshape
        kv = kv.unflatten(1, (-1, ratio))       # [b, s/r, r, d]
        score = score.unflatten(1, (-1, ratio))  # [b, s/r, r, d]

        # 对 overlap 模式（ratio==4），做 overlap_transform
        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

        # 加权求和：softmax 归一化后聚合
        kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [b, s/r, d]
```

### 2.3 Overlap 模式（仅 c4a 使用，ratio == 4）

对于 `ratio=4`，Compressor 使用**重叠窗口**（`self.overlap = compress_ratio == 4`）。其含义是：

每个压缩 token 实际上是 **8 个原始 token 的加权和**（stride=4），即连续两个压缩窗口各取一半组合：

```plain
原始 token: [t0, t1, t2, t3, t4, t5, t6, t7]
压缩 token 0: 来自 [t0, t1, t2, t3] 的后半段(d:) + [t0, t1, t2, t3, t4, t5, t6, t7] weight
压缩 token 1: 来自 [t0, t1, t2, t3] 的前半段(:d) + [t4, t5, t6, t7] 
```

`overlap_transform` 方法：

```python
def overlap_transform(self, tensor, value=0):
    """tensor: [b, s/r, r, 2d]"""
    b, s, _, _ = tensor.size()
    ratio, d = self.compress_ratio, self.head_dim
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]    # 后半部分
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]  # 前一窗口的前半部分（偏移一位）
    return new_tensor
```

这相当于 stride=4 的滑动窗口，每次步进 4 个原始 token，但窗口大小为 8。

### 2.4 Decode 增量更新（start_pos > 0，model.py:343-376）

```python
else:  # Decode 阶段（一次一个 token）
    should_compress = (start_pos + 1) % self.compress_ratio == 0
    score += self.ape[start_pos % ratio]  # 取对应的位置编码

    if overlap:
        # 写入状态缓冲区
        self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
        self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)

        if should_compress:  # 凑够 ratio 个了，执行压缩
            kv_state = torch.cat([
                self.kv_state[:bsz, :ratio, :d],     # 前半段（重叠部分）
                self.kv_state[:bsz, ratio:, d:],      # 后半段（当前窗口）
            ], dim=1)
            score_state = torch.cat([...], dim=1)
            kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            # 滑动状态缓冲区
            self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
    else:
        # 非 overlap（c128a）：简单 buffer
        self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
        self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
        if should_compress:
            kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
```

**Decode 阶段的核心技巧**：不重新计算所有压缩，而是维护一个小型状态缓冲区（`kv_state` 和 `score_state`），每次来一个新 token 就写入对应位置，当攒够 `ratio` 个后，一次性执行聚合。

### 2.5 压缩后的后处理（model.py:362-376）

```python
kv = self.norm(kv.to(dtype))              # RMSNorm
apply_rotary_emb(kv[..., -rd:], freqs_cis) # 对 rope 部分做 RoPE
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # 非rope部分FP8量化

# 写入 KV cache
if start_pos == 0:
    self.kv_cache[:bsz, :seqlen // ratio] = kv
else:
    self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

压缩后的 KV 被存入 `kv_cache`（位于 Attention 类管理的 `kv_cache[win:]` 区域，即 sliding window 部分之后的区域）。

------

## 三、两类压缩的对比

| 特性                    | c4a (ratio=4)                          | c128a (ratio=128)      |
| :---------------------- | :------------------------------------- | :--------------------- |
| 压缩倍数                | ~4x（因 overlap 实际为 8→1，stride=4） | ~128x（128→1，无重叠） |
| 覆盖范围                | 所有压缩层                             | 所有压缩层             |
| Indexer (DSA)           | ✅ 有，做 top-k 稀疏选择                | ❌ 无，全量 attend      |
| Overlap                 | ✅ 是                                   | ❌ 否                   |
| 1M token 后的压缩 KV 数 | ~250K                                  | ~8K                    |
| 主要目的                | 降低注意力计算量                       | 极致降低 KV cache 内存 |

------

## 四、DeepSeek Sparse Attention (DSA) 与 Indexer

### 4.1 Indexer 的原理

对于 c4a 层（ratio=4），即使压缩后 1M token 仍然有 ~250K 个压缩 KV。为了把注意力计算控制在可接受范围内，DSA 使用一个**轻量级 Indexer** 来挑选最重要的 top-k 压缩 token。

**Indexer 架构**（model.py:380-433）：

```python
class Indexer(nn.Module):
    def __init__(self, args, compress_ratio=4):
        self.n_heads = args.index_n_heads       # 64 个头（比主注意力少）
        self.head_dim = args.index_head_dim     # 128（比主注意力的 512 小）
        self.index_topk = args.index_topk       # 1024
        
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=bf16)
        
        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
```

**核心要点**：

1. **共享 Q 的低秩投影**：Indexer 使用与主 Attention 相同的 `qr`（`q_norm(wq_a(x))`），而不是重新计算 Query
2. **更小的 head_dim**：主注意力是 `n_heads=128, head_dim=512`，Indexer 是 `n_heads=64, head_dim=128`——计算量减少 8 倍
3. **独立的 Compressor**：Indexer 内部有自己的 Compressor（带 `rotate=True`，启用 Hadamard 旋转 + FP4 量化），用来压缩 KV 为 indexer 专用的小维度表示

### 4.2 Indexer 的前向计算（model.py:402-433）

```python
def forward(self, x, qr, start_pos, offset):
    # 1. Query 投影 + RoPE
    q = self.wq_b(qr)                   # [b,s] -> [b,s,64*128]
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = rotate_activation(q)            # Hadamard 旋转
    fp4_act_quant(q, fp4_block_size)    # FP4 量化（QAT）

    # 2. 压缩 KV（使用自己的 Compressor）
    self.compressor(x, start_pos)       # 结果写入 self.kv_cache

    # 3. 计算 similarity score
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    # 收集 KV 相似度
    index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos//ratio])
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    # 4. Causal mask
    if start_pos == 0:
        mask = torch.arange(seqlen//ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen+1).unsqueeze(1)//ratio
        index_score += torch.where(mask, float("-inf"), 0)

    # 5. Top-k 选择
    topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
    return topk_idxs
```

**数学等价形式**：

对于位置 `s` 的 query，其与位置 `t` 的压缩 KV 的匹配分数为：

```plain
score(s, t) = sum_over_heads( relu(Q_s · KV_t) * weight_s )
```

其中 `weight_s` 是 `weights_proj(x)` 的输出，作为一个**头级别的能量汇聚权重**（学出来的，表示每个头对最终选择的贡献度）。

选出的 top-k 索引被传递给 `sparse_attn` kernel，作为注意力计算的**稀疏索引集合**。

### 4.3 非 indexer 层的压缩索引（c128a）

对于 `ratio=128` 的层（无 Indexer），使用简单的 `get_compress_topk_idxs`（model.py:268-276）：

```python
def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
```

这本质上生成所有可达的压缩 token 索引（考虑了 causal masking），c128a 的注意力就是**全量密集注意力**作用于高度压缩后的 KV cache 上。

------

## 五、Sliding Window Attention (SWA)

### 5.1 原理

DeepSeek V4 对每个注意力层都维护一个大小为 128 的滑动窗口，query token 可以 attend 到它前面的 128 个**未压缩的原始 token**。

这解决了两个问题：

1. **局部性保护**：在压缩到达之前，query 可以获取精细的局部上下文
2. **压缩边界问题**：如果 query 刚好处于压缩块边界之前，滑动窗口可以覆盖尚未被压缩的信息

### 5.2 滑动窗口索引生成

```python
def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    # Prefill 阶段
    base = torch.arange(seqlen).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
    matrix = torch.where(matrix > base, -1, matrix)
    
    # Decode 阶段：使用循环缓冲区
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos+1, window_size), 
                           torch.arange(0, start_pos+1)], dim=0)
    # ...
```

**注意**：在 Decode 阶段，滑动窗口的 KV 使用**循环缓冲区**（`kv_cache[:bsz, start_pos % win]`），避免每次 decode 都复制大量数据。

### 5.3 滑动窗口的 KV 缓存管理（Attention.forward, model.py:517-533）

```python
# Prefill
if seqlen <= win:
    self.kv_cache[:bsz, :seqlen] = kv           # 直接存入
else:
    cutoff = seqlen % win
    # 只保留最后 win 个 token，但循环排列存储
    self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split(...)

# Decode
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

------

## 六、Sparse Attention Kernel（sparse_attn）

这是最底层的 CUDA kernel，用 TileLang 实现（kernel.py:276-368）。

### 6.1 原理

`sparse_attn` 接收：

- **Q**: `[b, s, h, d]` — query
- **KV**: `[b, n, d]` — KV cache（滑动窗口 + 压缩 KV 拼在一起）
- **topk_idxs**: `[b, s, topk]` — 每个 query 要 attend 的 KV 位置索引
- **attn_sink**: `[h]` — 可学习的 sink token bias

它执行的是 **FlashAttention 风格的在线 softmax**（running max + running sum），但 KV 不是整行读取，而是通过 `topk_idxs` **gather** 指定的位置：

```python
for each query position (bx, for batch by):
    q_shared = load Q[bx, by]
    acc_o = 0
    sum_exp = 0
    scores_max = -inf

    for each block of top-k indices:
        idxs = topk_idxs[by, bx, block*block_size + i]
        kv_shared = gather(kv[by, idxs[i]], for valid idxs)
        
        attn_scores = q_shared @ kv_shared^T * scale  # GEMM
        # 在线 softmax 稳定化
        new_max = max(attn_scores, dim=1)
        scores_scale = exp(old_max - new_max)
        acc_o *= scores_scale
        exp_scores = exp(attn_scores - new_max)
        sum_exp = sum_exp * scores_scale + sum(exp_scores)
        acc_o += exp_scores @ kv_shared
    
    # 加入 sink token（soft bias term）
    sum_exp += exp(attn_sink - scores_max)
    o = acc_o / sum_exp
```

**attn_sink 的作用**：这是一个可学习的偏置项，在 softmax 归一化时可以作为"默认"注意力权重，当 query attend 不到任何有效信息时，模型可以"sink"到这个项上，让输出更稳定（类似于 StreamingLLM 中的 attention sink 概念）。

### 6.2 配置信息

从 kernel 调用看（[model.py:528](http://model.py:528/)）：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

其中 `self.attn_sink = nn.Parameter(torch.empty(n_local_heads, dtype=torch.float32))` 绑定为每头一个标量。

------

## 七、Inverse RoPE：关键的正确性保障

在 DeepSeek V4 中有一个非常重要的设计细节（已经在 roadmap 中提到了）：

```python
# Attention.forward 中：
apply_rotary_emb(kv[..., -rd:], freqs_cis)   # 对 KV 的 rope 部分施加 RoPE

# ... attention 计算 ...

apply_rotary_emb(o[..., -rd:], freqs_cis, True)  # 对注意力输出的 rope 部分施加 INVERSE RoPE
```

**为什么需要 inverse RoPE？**

标准 MLA 中，position encoding 是施加在 query 和 key 上的。但在 DeepSeek V4 的压缩注意力中：

1. Q 和压缩后的 KV 各自都有自己的位置（RoPE 在各自的频率上编码）
2. 压缩 KV 的 RoPE 和 query 的 RoPE 是在**不同的 token 位置索引**上计算的
3. 注意力输出 `o` 是 `sum(softmax(score) * value)`——value 是带着 RoPE 的
4. **去掉 value 上的 RoPE（通过 inverse RoPE）**，使得输出不携带位置信息，从而可以在后续层继续进行位置编码

这个 inverse RoPE 步骤是**共享 K/V 向量（share key and value vectors）策略的核心**——因为如果不做 inverse RoPE，输出就会混入位置信息，导致后续层的位置编码混乱。

------

## 八、模型各层的完整数据流

### Prefill 阶段（首次处理大量 token）

```plain
输入 x [b, seqlen, dim]
  │
  ├─ HC Pre: x → [b, s, hc_mult, d] (Hyper-Connections 展开)
  │
  ├─ Attention:
  │   ├─ q = RMSNorm(wq_a(x))              # [b,s,qlora_rank]
  │   ├─ q = wq_b(q) → reshape → RoPE      # [b,s,n_heads,head_dim]
  │   ├─ kv = kv_norm(wkv(x))              # [b,s,head_dim]
  │   ├─ RoPE(kv[..., -rd:])               # 对 rope 部分编码
  │   │
  │   ├─ 滑动窗口索引: get_window_topk_idxs(128)
  │   │   → 每个 query 可 attend 前 128 个未压缩 token
  │   │
  │   ├─ 如果 compress_ratio != 0:
  │   │   ├─ [c4a] Indexer(x, qr) → top-1024 压缩 KV 索引（DSA）
  │   │   ├─ [c128a] get_compress_topk_idxs(128) → 所有可达压缩 KV
  │   │   └─ Compressor(x, start_pos) → 写入 kv_cache[win:]
  │   │
  │   ├─ topk_idxs = [SWA 索引] + [压缩索引]  # 合并
  │   ├─ kv = [滑动窗口 KV] + [压缩 KV]       # 合并到连续张量
  │   │
  │   ├─ sparse_attn(q, kv, attn_sink, topk_idxs)
  │   │   → gather 指定索引 → 在线 softmax
  │   │
  │   └─ Inverse RoPE(o)                     # 去除位置编码
  │
  ├─ HC Post + MoE FFN
  │   └─ Sinkhorn 归一化生成 pre/post/comb 权重
  │
  └─ 输出 [b, s, hc_mult, d]
```

### Decode 阶段（逐 token 生成）

```plain
输入 x [b, 1, dim]
  │
  ├─ q = 同上（只对当前 1 个 token）          # [b,1,n_heads,head_dim]
  ├─ kv = 同上（只对当前 1 个 token）          # [b,1,head_dim]
  │
  ├─ 滑动窗口: 写入 kv_cache[win][start_pos % 128]
  ├─ 压缩: 写入 Compressor 状态 buffer
  │
  ├─ topk_idxs = 
  │   ├─ 滑动窗口: 循环缓冲区中所有有值的索引（最多 128 个）
  │   └─ 压缩: 所有已生成的压缩 token（或 c4a 的 top-1024）
  │
  ├─ sparse_attn(q, kv_cache, attn_sink, topk_idxs)
  │
  └─ Inverse RoPE → HC Post → MoE
```

### 各层压缩策略差异总结

| 层索引 | compress_ratio | 压缩类型 | DSA Indexer | 压缩后 KV 数 (1M ctx) | 每次注意力的 KV 数 |
| :----- | :------------- | :------- | :---------- | :-------------------- | :----------------- |
| 0      | 128            | c128a    | ❌           | ~8K                   | 128(SW) + 全部8K   |
| 1      | 128            | c128a    | ❌           | ~8K                   | 128 + 全部8K       |
| 2      | 4              | c4a      | ✅           | ~250K                 | 128 + top-1024     |
| 3      | 128            | c128a    | ❌           | ~8K                   | 128 + 全部8K       |
| 4      | 4              | c4a      | ✅           | ~250K                 | 128 + top-1024     |
| …      | …              | …        | …           | …                     | …                  |
| 60     | 4              | c4a      | ✅           | ~250K                 | 128 + top-1024     |
| 61     | 0              | SWA only | ❌           | 0                     | 128(SW)            |

### 性能数据

根据论文中的描述（README 提及）：

- **1M token 推理**：DeepSeek V4 Pro 仅需 DeepSeek V3.2 的 **27% 单 token FLOPs**
- **KV cache 压缩到 10%**：相比标准 MHA 或之前版本的 MLA
- 这些收益来自三个层面：
  - c128a 层极致压缩 KV cache 内存（128x）
  - c4a + DSA 层大幅降低注意力计算量（只 attend top-1024 而非全部 250K）
  - SWA 用 tiny 窗口保障局部性

------

## 九、总结

DeepSeek V4 的混合注意力架构通过**分层压缩策略**优雅地解决了百万级 token 上下文的两大瓶颈：

1. **KV cache 爆炸** → 所有非滑动窗口层使用 c4a（4x 压缩）或 c128a（128x 压缩）
2. **注意力计算量过大** → c4a 层配合 DSA/Indexer 只 attend top-1024（稀疏注意力）；c128a 层虽然全量 attend，但 KV 只有 ~8K 个

再加上 Attn Sink（学习到的默认注意力偏置）、Inverse RoPE（共享 KV 的正确性保障）、以及 TileLang 高性能 CUDA kernel（FP8/FP4 量化 + sparse attention 索引 gather），这套设计在工程和算法层面都非常精巧。