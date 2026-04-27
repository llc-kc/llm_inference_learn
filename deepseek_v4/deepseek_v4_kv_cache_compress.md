# DeepSeek V4 KV Cache 压缩原理深度解析

## 概览：从上层视角理解三套压缩机制

DeepSeek V4 的注意力机制在标准 MLA（Multi-head Latent Attention）之上，增加了**三层 KV cache 压缩**机制，分别对应 **SWA（Sliding Window Attention）、C4A（Compress 4：1 Attention）和 C128A（Compress 128：1 Attention）**。每层压缩的目标和粒度不同，协同工作来让百万级 token 推理成为可能。

```plain
                    图中数字含义
  ┌────────────────────────────────────────────────────────────┐
  │  query token (s)                                          │
  │    │                                                      │
  │    ├── SWA (window=128) → 最近 128 个原始 token           │
  │    │     kv_cache 存储: 128 个未压缩的 token              │
  │    │                                                      │
  │    ├── C4A (ratio=4) → 最相关 ~512 个压缩 token          │
  │    │     1 个压缩 token = 8 个原始 token 的加权求和       │
  │    │     stride=4 → 步长 4，有 overlap                    │
  │    │     kv_cache 存储压缩后: 原始/4                       │
  │    │                                                      │
  │    └── C128A (ratio=128) → 最相关 ~512 个压缩 token      │
  │          1 个压缩 token = 128 个原始 token 的加权求和     │
  │          stride=128 → 无 overlap                          │
  │          kv_cache 存储压缩后: 原始/128                     │
  │                                                           │
  │  Sparse Attention: 从 C4A/C128A KV 中 top-k 选出 ~512 个 │
  │  参与 attention 计算 (DSA: DeepSeek Sparse Attention)     │
  └────────────────────────────────────────────────────────────┘
```

控制每个 layer 启用哪种压缩的参数是 `compress_ratios`，7 层定义如下：

```python
compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
```

这是一个 **per-layer 的配置数组**——每一层独立选择压缩策略：

| Layer | ratio | 含义                          | 压缩倍率 |
| :---- | :---- | :---------------------------- | :------- |
| 0     | 0     | 纯 Sliding Window（SWA only） | 不压缩   |
| 1     | 0     | 纯 Sliding Window（SWA only） | 不压缩   |
| 2     | 4     | **C4A** + SWA                 | ~4x      |
| 3     | 128   | **C128A** + SWA               | ~128x    |
| 4     | 4     | **C4A** + SWA                 | ~4x      |
| 5     | 128   | **C128A** + SWA               | ~128x    |
| 6     | 4     | **C4A** + SWA                 | ~4x      |
| 7     | 0     | 纯 Sliding Window（SWA only） | 不压缩   |

> 注意：这里虽然 index 是 7 但 [model.py](http://model.py/) 中 `n_layers=7`，所以实际 0-6 共 7 层，第 0/1 是纯 SWA，第 2/4/6 是 C4A + SWA，第 3/5 是 C128A + SWA。

------

## 一、SWA — Sliding Window Attention（局部注意力层）

### 原理

SWA 保证 query token 总能关注到**最近的 `window_size=128` 个原始 token**。这是所有层的"兜底"机制——即使在有压缩的层，非压缩的滑动窗口也永远存在。

### 代码实现

```python
# Attention.__init__ (model.py:285)
self.window_size = args.window_size  # = 128
win = self.window_size

# kv_cache 总大小 = window_size + 压缩部分
kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
# 压缩部分存储在 kv_cache[:, win:] 位置
```

关键点：**`kv_cache` 的第一个 `win`（128）个槽位专门存放滑动窗口的原始 KV**。

### 窗口索引的生成

```python
@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        # 滚动模式：从 start_pos+1 到 window_size，再 wrap around 到 start_pos+1
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torque.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        # 前缀填充模式：前 start_pos+1 个位置有效，其余为 -1（masked）
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        # decode 阶段：每个位置生成自己的窗口索引
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
```

**这段代码的精妙之处**：`get_window_topk_idxs` 用 `@lru_cache` 缓存，因为给定相同的 `window_size, bsz, seqlen, start_pos`，结果永远是**确定性的**——无需实时计算。

**三种模式的区分**：

1. **`start_pos >= window_size - 1`（滚动填充后）**：窗口已经填满，新的 KV 写入 `start_pos % win` 位置。窗口索引是一个**循环排列**：`[start_pos+1, start_pos+2, ..., 127, 0, 1, ..., start_pos]`。
2. **`start_pos > 0`（前缀填充中）**：窗口还没填满，前 `start_pos+1` 个位置有效，后面的位置塞 `-1`（sparse_attn 见到 `-1` 索引会填 0，等效于 mask 掉）。
3. **`start_pos == 0`（prefill 阶段）**：为 seqlen 个 query 中的每个位置生成其合法的窗口索引——`[max(0, t-window_size+1), ... t]`，超出范围的塞 `-1`。

### SWA KV 的写入

```python
# prefill (start_pos == 0)
if seqlen <= win:
    self.kv_cache[:bsz, :seqlen] = kv          # 全部放前面
else:
    cutoff = seqlen % win
    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split(...)
    # 循环填充：从 cutoff 位置开始放，wrapped 到前面

# decode (start_pos > 0)
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)  # 循环覆盖
```

### 关键洞察

SWA 存储的是**未压缩的原始 KV**，这意味着 128 窗口内的 prefix 信息完全保真。对于需要精确匹配的局部模式（如代码中的变量引用、自然语言中指代消解），SWA 提供了无损失的注意力路径。

------

## 二、Compressor 类 — C4A 和 C128A 的共享实现骨架

`Compressor` 是 **C4A 和 C128A 共同的压缩引擎**。它的核心思想是：**将连续 N 个 token 的 KV 通过一个可学习的门控加权求和，合并成一个 token**，从而大幅压缩 KV cache。

### 构造函数解析

```python
class Compressor(nn.Module):
    def __init__(self, args, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim      # = 64 (只有最后 64 维被 RoPE 处理)
        self.nope_head_dim = head_dim - args.rope_head_dim  # = 448 (非 RoPE 部分)
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4   # ⭐ C4A 有 overlap，C128A 无
        self.rotate = rotate                 # Indexer 专用的 Compressor 会 rotate
        coff = 1 + self.overlap              # C4A: coff=2, C128A: coff=1
        ...
```

**⭐ 核心设计差异**：

- **C4A（ratio=4）**：`overlap=True`，`coff=2`——意味着每个压缩 token 的维度翻倍（2×head_dim），一半用于正常的压缩窗口，一半用于**跨窗口的重叠信息**。
- **C128A（ratio=128）**：`overlap=False`，`coff=1`——每个压缩 token 的维度等于 head_dim，无重叠。

### 学习型压缩的参数

```python
self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
self.norm = RMSNorm(self.head_dim, args.norm_eps)
```

这里有三个可学习组件：

1. **`wkv`**：将输入 x（dim=4096）投影到 KV 压缩空间（2×head_dim 或 1×head_dim）
2. **`wgate`**：计算门控分数——决定每个 token 在压缩加权中的权重
3. **`ape`**（Absolute Position Embedding）：`[compress_ratio, coff*head_dim]`——位置嵌入，让门控系统知道当前 token 在压缩窗口内的**相对位置**

### 状态缓冲区（解码阶段的增量状态）

```python
self.register_buffer("kv_state", torch.zeros(batch_size, coff * compress_ratio, coff * head_dim))
self.register_buffer("score_state", torch.full((batch_size, coff * compress_ratio, coff * head_dim), -inf))
```

这是**流式推理的精华**——解码阶段每次只来 1 个新 token，不能像 prefill 那样一次处理整个窗口。`kv_state` 和 `score_state` 就是用来暂存未满一个窗口的 KV 和分数。

------

## 三、C4A（ratio=4）— 带 Overlap 的局部压缩

### 整体思路

C4A 的压缩比是 4:1。但注意：**每个压缩 token 是 8 个原始 token 的加权和，stride=4，有 overlap**。

```plain
  原始 token: 0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 ...
               \  \  \  \  /  /  /  /
                \  \  \  \/  /  /  /          C4A stride=4, window=8
                 \  \  \ /\  /  /  /
                  \  \  \/  \/  /  /
                   \  \/  \  \/  /
                    \ /\  /\  /\/
                     X────X────X──── ...
              压缩 token 0  压缩 token 1  压缩 token 2
              (覆盖 token 0-7) (覆盖 token 4-11) (覆盖 token 8-15)
```

每个压缩 token 本质上是一个**滑动窗口加权平均**，窗口大小 8，步长 4，重叠 4 个 token。这样做的目的是：

- 提供更平滑的压缩边界
- 避免信息在压缩边界处断裂

### Prefill 阶段的 C4A 实现

```python
if start_pos == 0:
    should_compress = seqlen >= ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    offset = ratio if overlap else 0  # C4A: offset = 4

    # Step 1：保存最后一个窗口给 kv_state/score_state（用于下个压缩块的 overlap）
    if overlap and cutoff >= ratio:
        self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
        self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape

    # Step 2：处理余数部分（无法构成完整压缩块的尾段）
    if remainder > 0:
        kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
        self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
        score = score[:, :cutoff]

    # Step 3：将完整 KV reshape 为 [batch, num_blocks, ratio, coff*head_dim]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape

    # ⭐ Step 4：C4A 独有——overlap 转换
    if overlap:
        kv = self.overlap_transform(kv, 0)               # pad 0 for overlapping
        score = self.overlap_transform(score, float("-inf"))  # pad -inf for overlapping

    # Step 5：门控加权求和 → 压缩
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

### overlap_transform 的精妙设计

```python
def overlap_transform(self, tensor: torch.Tensor, value=0):
    # tensor: [b, s, ratio, 2*d] — 前 d 维是正常压缩，后 d 维是 overlap 部分
    b, s, _, _ = tensor.size()
    ratio, d = self.compress_ratio, self.head_dim
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]       # 后 d 维 → 新 tensor 的后 ratio 个位置
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]    # ⭐ 前一个块的正常部分 → 当前块的 overlap 部分
    return new_tensor
```

这里 `wkv` 的输出是 `coff*head_dim = 2*head_dim`。前 `head_dim` 维是"正常窗口"信息，后 `head_dim` 维是"重叠窗口"信息。

```plain
overlap_transform 的视觉效果：

输入 tensor [b, s, 4, 2d]：

块0: [───d───│───d───]   块1: [───d───│───d───]   块2: [───d───│───d───]
      正常 0  重叠 0         正常 1  重叠 1         正常 2  重叠 2

输出 new_tensor [b, s, 8, d]：

块0: [   0   │  0   │  0   │  0   │  重叠0│  正常0│  正常0│  正常0]  ← 前 4 个为 value
      └───────── 前 ratio 个 ──────┘└───────── 后 ratio 个 ───────┘

块1: [  重叠0│  正常0│  正常0│  正常0│  重叠1│  正常1│  正常1│  正常1]
      └── 来自 tensor[:, 0, :, :d] ──┘└── 来自 tensor[:, 1, :, d:] ─┘

块2: [  重叠1│  正常1│  正常1│  正常1│  重叠2│  正常2│  正常2│  正常2]
      └── 来自 tensor[:, 1, :, :d] ──┘└── 来自 tensor[:, 2, :, d:] ─┘
```

这样**每个压缩 token 实际看到 8 个原始 token 的信息**（前一个块的最后 4 个 + 本块的全部 4 个），从而实现了 stride=4, window=8 的 overlapping 压缩。门控系统通过 `softmax(dim=2)` 来决定这 8 个位置中每个位置的权重。

### Decode 阶段的 C4A 实现

```python
else:  # start_pos > 0 (decode phase)
    should_compress = (start_pos + 1) % self.compress_ratio == 0  # 每 4 个 token 压缩一次
    score += self.ape[start_pos % ratio]  # 加上位置编码

    if overlap:  # C4A 路径
        # Step 1: 保存当前 token 到状态缓冲区
        self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
        self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)

        if should_compress:
            # Step 2: 拼接重叠窗口 + 当前窗口
            kv_state = torch.cat([
                self.kv_state[:bsz, :ratio, :d],      # 前一个窗口的后半段（重叠部分）
                self.kv_state[:bsz, ratio:, d:]       # 当前窗口的前半段（正常部分）
            ], dim=1)
            score_state = torch.cat([
                self.score_state[:bsz, :ratio, :d],
                self.score_state[:bsz, ratio:, d:]
            ], dim=1)

            # Step 3: 门控加权求和
            kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

            # Step 4: 搬运状态——将当前窗口 shift 到重叠位置
            self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
            self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
```

解码阶段 `kv_state` 和 `score_state` 的布局：

```plain
kv_state [b, 8, 2d]  (coff=2, compress_ratio=4, coff*compress_ratio=8)
┌─────────────────────────────────────────────────────────┐
│ 0..3 (重叠部分) │ 4..7 (当前窗口部分) │
│ [0..3, :d] = 上一个窗口的后半段   │
│ [4..7, d:] = 当前窗口的 4 个 token │
└─────────────────────────────────────────────────────────┘
```

每次压缩完成后，将 `kv_state[:, 4:7]`（当前窗口）搬到 `kv_state[:, 0:3]`（成为下一个压缩块的重叠部分），实现了状态的无缝接力。

------

## 四、C128A（ratio=128）— 全局语义压缩

### 整体思路

C128A 的压缩比是 128:1，其**核心哲学**是：连续的 128 个 token 通常共享一个粗粒度的语义主题（一个段落、一个代码块、一段论证），可以用单个压缩 token 来表征。这相对于 C4A 的局部模式捕捉，C128A 更像**语义摘要**。

```plain
原始 tokens: 0─1─2─...─127─128─129─...─255─256─257─...─383 ...
              │                  │                    │
              ▼                  ▼                    ▼
         compressed[0]    compressed[1]          compressed[2]
         加权和 128 tokens   加权和 128 tokens      加权和 128 tokens
```

### C128A 与 C4A 的区别

| 特性                | C4A (ratio=4)                                      | C128A (ratio=128)                |
| :------------------ | :------------------------------------------------- | :------------------------------- |
| `overlap`           | True                                               | False                            |
| `coff`              | 2                                                  | 1                                |
| 每个压缩 token 覆盖 | 8 个原始 token（overlap）                          | 128 个原始 token                 |
| 维度                | 2×head_dim（一半 overlap 用）                      | head_dim                         |
| 位置编码            | ape: [4, 2×head_dim]                               | ape: [128, head_dim]             |
| **Indexer**         | **有**（C4A 层配置了 Indexer 做 sparse attention） | **无**（C128A 层直接用固定索引） |

### Prefill 阶段 C128A 的代码路径

```python
if start_pos == 0:
    # ... 与 C4A 一样的逻辑结构，但 overlap=False
    offset = 0  # C128A: 没有 offset

    if overlap and cutoff >= ratio:  # C128A 永远不会进入
        ...

    if remainder > 0:
        kv, self.kv_state[:bsz, 0 : remainder] = kv.split([cutoff, remainder], dim=1)
        self.score_state[:bsz, 0 : remainder] = score[:, cutoff:] + self.ape[:remainder]
        score = score[:, :cutoff]

    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape

    # overlap=False → 没有 overlap_transform
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

因为 `overlap=False`，`coff=1`，门控系统只需在 128 个 token 上做 softmax 加权。每个压缩 token 就是 128 个 token 的带权摘要。

### Decode 阶段 C128A 的代码路径

```python
else:  # C128A 走无 overlap 分支
    should_compress = (start_pos + 1) % self.compress_ratio == 0  # 每 128 token 压缩一次
    score += self.ape[start_pos % ratio]

    # 无 overlap 路径
    self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
    self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)

    if should_compress:
        kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
```

简单直接——每来 128 个新 token，在 `kv_state`（128 槽位）中累积，满了就做一次 128→1 的软平均。没有 C4A 的状态搬运，因为下一个 128 块完全独立于上一个。

------

## 五、压缩后的处理：归一化 + RoPE + 量化

不论 C4A 还是 C128A，压缩完成后有一段统一的处理：

```python
kv = self.norm(kv.to(dtype))  # RMSNorm 归一化

if start_pos == 0:
    freqs_cis = self.freqs_cis[:cutoff:ratio]  # 每个压缩 token 使用其对应位置的 RoPE
else:
    freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)  # 最后位置的 RoPE

apply_rotary_emb(kv[..., -rd:], freqs_cis)  # 只对最后 rope_head_dim=64 维应用 RoPE

if self.rotate:
    kv = rotate_activation(kv)          # Indexer 专用：Hadamard 旋转
    fp4_act_quant(kv, ...)              # FP4 量化
else:
    act_quant(kv[..., :-rd], ...)       # 非 RoPE 部分做 FP8 量化（RoPE 保持 BF16）
```

重要细节：

- **RoPE 只应用到最后 64 维**（`rope_head_dim=64`），前 448 维（`nope_head_dim=448`）不带位置编码
- **非 RoPE 部分做 FP8 量化**，RoPE 部分保持 BF16——因为位置信息需要高精度
- **Indexer 的压缩**额外做了 FP4 量化用于粗筛评分

------

## 六、Sparse Attention — 如何从压缩 cache 中选 top-k

### 对 C4A 层：使用 Indexer 进行稀疏选优

C4A 层有 Indexer（ratio=4），因为即使经过 4 倍压缩，百万 token 仍有 25 万压缩 token——全部参与 attention 太昂贵。Indexer 负责从中选出最重要的 `index_topk=512` 个。

```python
class Indexer(nn.Module):
    def __init__(self, args, compress_ratio=4):
        self.index_topk = args.index_topk  # = 512
        self.n_heads = args.index_n_heads   # = 64
        self.head_dim = args.index_head_dim  # = 128 (比主注意力的 512 更小！)

        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
        # ↑ Indexer 有自己的轻量 Compressor（head_dim=128, 带 Hadamard 旋转+FP4量化）
```

**Indexer 的工作流程**：

1. **用自己的轻量 Compressor** 压缩 KV（维度更小：128 vs 512，FP4 量化）
2. **用自己的 Q 投影**（`wq_b`）用低秩 Q（`q_lora_rank=1024`）投影得到 query
3. **计算评分**：`q @ kv_cache^T` 得到每个压缩 token 的分数
4. **用权重投影**（`weights_proj`）加权合并多个 head 的分数
5. **取 top-512** 作为稀疏注意力索引

```python
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

**为什么 Indexer 的 head_dim 是 128 而不是 512？**——因为这是一个粗筛过程，不需要主注意力那么高的精度。128 维 + FP4 量化足够区分哪些压缩 token 值得关注。

### 对 C128A 层：使用固定步长索引

C128A 层没有 Indexer，而使用 `get_compress_topk_idxs` 函数生成固定索引：

```python
@lru_cache(2)
def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        # decode：当前压缩 token 之前的全部压缩 token
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        # prefill：每个位置能看到其之前所有完整的压缩块
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
```

**为什么 C128A 不需要 Indexer？**——因为 C128A 的压缩比是 128:1，1M token 只有 ~8K 压缩 token，完全可以全部参与 attention 计算（topk=512 的限制实际上远小于总压缩数，但这里直接用固定索引选择了所有压缩 token 中符合条件的部分）。

### 最终索引拼接

```python
# Attention.forward 中
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)  # SWA 索引

if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win  # 压缩 KV 在 kv_cache 中从 win 位置开始
    if self.indexer is not None:  # C4A
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:  # C128A
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

**SWA 索引（0…win-1）+ 压缩索引（win…win+compressed_len）拼接成最终的稀疏索引**，传入 `sparse_attn` kernel。

------

## 七、sparse_attn kernel — TileLang 实现的核心

```python
@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
```

这个 CUDA kernel 通过 TileLang 生成，实现了**带 online softmax 的稀疏注意力**：

1. **按 topk_idxs 从 KV cache 中 gather** 出当前 query 需要关注的 KV
2. **FlashAttention 风格的在线 softmax**：维护 `scores_max`（当前最大值）和 `sum_exp`（exp 和），逐块更新
3. **attn_sink 偏置**：一个可学习的 bias（`nn.Parameter`），作为 attention 的"先验"占位，确保数值稳定性

```python
# kernel 伪码逻辑
acc_o = 0
sum_exp = 0
scores_max = -inf

for each block of top-k KV:
    gather kv by topk_idxs
    scores = q @ kv^T * scale       # [h, block_size]
    scores_max_prev = scores_max
    scores_max = max(scores_max, max(scores, dim=1))
    scores_scale = exp(scores_max_prev - scores_max)
    sum_exp = sum_exp * scores_scale + sum(exp(scores - scores_max))
    acc_o = acc_o * scores_scale + exp(scores - scores_max) @ kv

sum_exp += exp(attn_sink - scores_max)  # 添加 sink 偏置
o = acc_o / sum_exp
```

------

## 八、总结：三层压缩的协同工作

| 组件              | SWA          | C4A                               | C128A                |
| :---------------- | :----------- | :-------------------------------- | :------------------- |
| **压缩比**        | 1x (不压缩)  | ~4x                               | ~128x                |
| **粒度**          | token 级     | 短语/子句级                       | 段落/块级            |
| **窗口**          | 128 tokens   | 8 tokens (overlap)                | 128 tokens           |
| **索引策略**      | 循环窗口     | Indexer 学习评分 top-512          | 固定位置索引         |
| **KV cache 存储** | 原始 BF16    | 压缩后 FP8                        | 压缩后 FP8           |
| **作用**          | 局部精确匹配 | 中等距离语义                      | 全局语义摘要         |
| **1M tokens 时**  | 128 个       | ~250K 个（经筛选 512 个参与计算） | ~8K 个（经固定选择） |

**完整的数据流**：

```plain
输入 x → MLA: wkv(norm) + RoPE + FP8量化
         │
         ├─→ SWA: 原始 KV → kv_cache[:, 0:128]
         │
         ├─→ Compressor: gated weighted sum → 压缩 KV
         │    ├─ C4A: 8→1, overlap, indexer 筛选 top-512
         │    │   → kv_cache[:, 128:128+total/4]
         │    └─ C128A: 128→1, no overlap, 固定选取
         │        → kv_cache[:, 128:128+total/128]
         │
         └─→ Sparse Attention: 
              topk_idxs = SWA_idxs + (indexer_idxs | fixed_idxs)
              gather → online softmax → output
              
         output → inverse RoPE → 分组低秩 O 投影
```

**最终效果**：对于 1M token 的推理，每层 attention 只需要计算：

- SWA: 128 个原始 KV（精确局部信息）
- C4A: ~512 个压缩 KV（由 Indexer 挑选的最相关中层语义）
- C128A: 数十到数百个压缩 KV（全局语义锚点）
- **总计不到 1000 个 KV 参与计算**，而非 1M 个，实现了数量和计算量的双重压缩。