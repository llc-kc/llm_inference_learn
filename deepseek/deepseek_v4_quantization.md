Deepseek v4 4bit量化

------

下面从 **架构概览 → 4-bit Indexer → MoE FP4 量化** 三个层面进行详细介绍。

## 一、架构概览

来自 `config.json` 的核心参数（真实模型，非测试用的小规模）：

| 参数                          | 值                             |
| :---------------------------- | :----------------------------- |
| `hidden_size`                 | 7168                           |
| `num_hidden_layers`           | 61                             |
| `num_attention_heads`         | 128                            |
| `head_dim`                    | 512                            |
| `n_routed_experts`            | 384                            |
| `n_activated_experts` (top-k) | 6                              |
| `n_shared_experts`            | 1                              |
| `moe_intermediate_size`       | 3072                           |
| `max_position_embeddings`     | 1,048,576 (1M tokens)          |
| `sliding_window`              | 128                            |
| `q_lora_rank`                 | 1536                           |
| `o_lora_rank`                 | 1024                           |
| `o_groups`                    | 16                             |
| `hc_mult`                     | 4 (Hyper-Connections 的多重态) |
| `index_topk`                  | 1024                           |
| `compress_ratios`             | 61个layer的压缩率数组          |

模型核心组件：

- **MLA (Multi-head Latent Attention)** — 低秩 Q/KV 投影 + 滑动窗口
- **KV Compression** — 对长距离 KV 做门控池化压缩
- **4-bit Indexer** — 从压缩 KV 中智能选取 top-k 位置
- **MoE** — 384个路由专家 + 1个共享专家，支持 FP4 权重量化
- **Hyper-Connections (HC)** — 替代残差连接，维护 `hc_mult` 份隐藏状态副本，通过 Sinkhorn 平衡学习混合权重

------

## 二、4-bit Indexer（4位索引器）

### 2.1 设计目标

传统稀疏注意力要么用滑动窗口（固定局部性），要么用某种全局均匀采样。Indexer 的目标是**学习性地从压缩后的 KV 中挑选出对当前 token 最重要的 top-k 个压缩位置**，用于长距离稀疏注意力。

### 2.2 代码位置

`inference/model.py`，`Indexer` 类（第 380-433 行）。

### 2.3 详细结构

```plain
text复制Indexer:
  ├── compressor: Compressor(head_dim=128, rotate=True)  ← 独立的压缩器
  ├── wq_b: ColumnParallelLinear(q_lora_rank, 64*128)    ← 从 qr (低秩Q) 投影到 indexer 的 query
  ├── weights_proj: ColumnParallelLinear(dim, 64)         ← 每个 head 的权重标量
  ├── kv_cache: [B, L//4, 128]                            ← 索引器自己的压缩 KV 缓存
  ├── index_topk: 1024
  └── softmax_scale: d^-0.5
```

关键参数：`index_n_heads=64`, `index_head_dim=128`, `index_topk=1024`。

注意：**Indexer 的 head_dim=128 ≠ 主 Attention 的 head_dim=512**。Indexer 使用更小的 128 维空间进行评分，大幅减少计算量。

### 2.4 前向计算流程

```plain
text复制forward(x, qr, start_pos, offset):
  ├── q = wq_b(qr)                          # [B,S,64*128] ← 从低秩qr投影
  │   q = unflatten → [B,S,64,128]          # 64 local heads
  │   apply_rotary_emb(q[..., -64:])         # 对最后64维做RoPE
  │   q = rotate_activation(q)               # Hadamard变换
  │   q = fp4_act_quant(q, inplace=True)     # ⭐ QAT模拟：FP4量化和反量化
  │
  ├── compressor(x, start_pos)               # ⭐ 构建压缩KV缓存
  │   compress_ratio=4, 有Hadamard旋转
  │   输出存储在 self.kv_cache: [B, L//4, 128]
  │   ⭐ 同样经过 fp4_act_quant 模拟
  │
  ├── weights = weights_proj(x)              # [B,S,64]
  │   * (softmax_scale * n_heads^-0.5)        # 每个(head, position)的权重标量
  │
  ├── index_score = einsum("bshd,btd->bsht", q, kv_cache)
  │   # [B,S,64, L//4] — 每个query位置×每个head ×所有压缩KV位置
  │   index_score = relu(index_score) * weights.unsqueeze(-1)
  │   index_score = sum(dim=2)               # [B,S, L//4] — 跨head聚合
  │
  └── topk_idxs = topk(index_score, k=min(1024, L//4))
      return topk_idxs                        # [B,S,1024]
```

### 2.5 核心创新点

**a) 4-bit 量化模拟推理（QAT 对齐）**

Indexer 内部进行了大量的 `fp4_act_quant(x, block_size=32, inplace=True)` 调用。这并不是为了节省显存，而是**模拟量化感知训练 (QAT)** 的效果——在推理时对 activation 做"先量化为 FP4、再反量化为 BF16"的原地操作，使得数值行为与训练时看到的一致。第 414-416 行：

```python
python复制q = rotate_activation(q)
fp4_act_quant(q, fp4_block_size, True)   # FP4 quant + dequant
self.compressor(x, start_pos)             # compressor内部也在rotate后调用了fp4_act_quant
```

**b) 带权重的评分机制**

Indexer 不是简单地用 query 和 KV 做点积就完事。它同时引入了 `weights_proj` 层，产生每个(head, position)位置的可学习权重，然后对每个 head 的评分做 ReLU 后乘以权重再跨 head 求和。这起到了类似注意力池化的效果，让不同 head 对最终的 top-k 选择有不同的贡献度。

**c) 独立的小维度压缩 KV**

Indexer 维护自己的 Compressor（`head_dim=128`），不同于主 Attention 的 Compressor（`head_dim=512`）。这是因为：

- Indexer 只需要足够的信息来**评分位置重要性**，不需要完整的语义表征
- 128 维的 key/value 显著降低了 `[B,S,64,128]×[B,L//4,128]` 评分计算的 FLOPs

**d) 因果掩码**

```python
python复制mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen+1).unsqueeze(1) // ratio
index_score += torch.where(mask, float("-inf"), 0)
```

确保每个 token 只能 attend 到它之前的压缩位置，不能看到未来的 token。

------

## 三、MoE FP4 量化算法

### 3.1 整体架构

```plain
text复制MoE:
  ├── Gate: 384个专家的路由门控
  ├── experts: 384个 (routed) Expert，每个是 SwiGLU FFN
  │   └── Expert: w1 (dim→moe_inter), w2 (moe_inter→dim), w3 (dim→moe_inter)
  └── shared_expert: 1个共享 Expert（全量访问）
```

- `n_routed_experts=384`, `n_activated_experts=6` (top-6)

- `moe_intermediate_size=3072`, `hidden_size=7168`

- 路由函数：`sqrtsoftplus`：

  ```python
  python复制scores = F.softplus(scores).sqrt()   # 默认score_func
  ```

- 路由缩放因子：`route_scale=2.5`

- **当 `expert_dtype="fp4"` 时，专家的 w1/w2/w3 权重以 FP4 格式存储**

### 3.2 FP4 权重存储格式

**数据格式**：`torch.float4_e2m1fn_x2` — 每8位(1 byte)存两个 FP4 值，packed 沿 K 维。

**量化粒度**：**per-32 元素一个缩放因子**（`fp4_block_size=32`），存储在 `torch.float8_e8m0fnu`（E8M0，纯指数格式）的张量中。

```plain
text复制Weight:  [out_features, in_features//2]   (float4_e2m1fn_x2)
Scale:   [out_features, in_features//32]  (float8_e8m0fnu)
```

每32个连续的 FP4 元素共享一个 E8M0 缩放因子。

**FP4 数值表**（来自 `convert.py` 第 11-14 行）：

```plain
text复制FP4_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    # 正数 (e2m1)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # 负数
]
```

FP4 (e2m1fn) 格式：1位符号 + 2位指数 + 1位尾数，所以精度很低但范围还算可以（±6.0）。

### 3.3 FP4 量化推理流程

推理时，一个 FP4 权重的 Expert 的前向传播：

```plain
text复制expert.forward(x):
  ├── gate = w1(x)      # FP8 activations × FP4 weight
  ├── up = w3(x)        # FP8 activations × FP4 weight
  ├── x = SiLU(gate) * up
  └── output = w2(x)    # FP8 activations × FP4 weight (反向)
```

每个 Linear 层通过 `linear()` 函数（`model.py` 第 108-120 行）分发：

```python
python复制def linear(x, weight, bias=None):
    if weight.dtype == torch.float4_e2m1fn_x2:    # ← FP4权重的分支
        x, s = act_quant(x, block_size, ...)       # 激活量化为FP8
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
```

流程分两步：

**Step 1：激活值 FP8 量化** (`act_quant`)

- 对 activation tensor 做 **block-wise FP8 量化**，`block_size=128`
- 量化方式：每个 128 元素的块求绝对最大值 → `scale = amax / 448`，然后将每个元素除以 scale 并 clamp 到 FP8 范围 [-448, 448]
- 当 `scale_fmt="ue8m0"` 时，scale 会被 round 到 2 的幂次（MXFP 风格），使得后续反量化更快
- 最终输出：x_fp8: `[..., K]` 和 s: `[..., K/128]`

**Step 2：FP4×FP8 GEMM** (`fp4_gemm`)

这是最核心的量化算法部分（`kernel.py` 第 441-536 行）。

```plain
text复制C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T
  
  A: per-128 act scale (FP8)
  B: per-32 weight scale (FP4, E8M0), stored as [N, K/2] (packed)
```

CUDA kernel 执行策略：

```plain
text复制对每个输出块 [block_M=32, block_N=128]:
  ├── 加载 A: [32, 32] FP8 (每次加载 block_K=32)
  ├── 加载 B: [128, 32] FP4
  │   └── FP4 → FP32 → FP8 逐元素转型
  ├── 加载A的scale: [32] → scale_a_frag (对应k//4, 因为act block=128而k step=32)
  ├── 加载B的scale: [128] → scale_b_frag (对应k)
  ├── GEMM: [32,32] @ [128,32]^T → [32,128] 在fp8精度
  └── 累加器: C += gemm_result * scale_a * scale_b
```

关键优化：

- 每个 block_K = 32，**正好等于 weight_group_size**，使得 per-32 的 weight scale 可以直接索引
- Act scale 是 per-128，每 4 个内循环共享一个 act scale（`n_sub = 128/32 = 4`）
- 使用**两个累加器** `C_local` 和 `C_local_accum` 分离 GEMM 累加和 scale 校正，提高精度

### 3.4 FP4 → FP8 转换期间的精度管理

在 `convert.py` 中有一个 `cast_e2m1fn_to_e4m3fn` 函数（第 17-52 行），用于将 FP4 权重无损转换为 FP8。算法：

```python
python复制# 1. 解包 packed FP4
low  = x & 0x0F     # 取低4位
high = (x >> 4) & 0x0F   # 取高4位
x = FP4_TABLE[low]  # 查表得到浮点值

# 2. 计算偏移量
# FP4最大值6.0, FP8最大值448
# MAX_OFFSET_BITS = 6  (6.0 * 2^6 = 384 < 448)
# scale_max_offset = per-128块的amax / 2^6
# offset = scale / scale_max_offset  → 每32个FP4元素有不同的偏移

# 3. 重整为 [bOut, bIn, 128, 128] 的块结构
# x * offset 后转换为 float8_e4m3fn
```

这使得可以在保持精度的前提下将 FP4 权重转换为 FP8 进行兼容部署。

### 3.5 量化对 Indexer 的特殊应用

Indexer 内部对 query 和压缩 KV 都做了 **FP4 量化模拟**（`fp4_act_quant` with `inplace=True`），但此处是"模拟"而非真实压缩存储：

- 目的：**保证推理时的数值分布与 QAT（量化感知训练）时完全一致**
- 做法：`quantize → dequantize` 原地做，数据仍然是 BF16，但已丢失了 FP4 精度导致的信息
- 第 415-416 行：`q = rotate_activation(q)` → `fp4_act_quant(q, fp4_block_size, True)`
- 第 370-371 行（compressor内）：`kv = rotate_activation(kv)` → `fp4_act_quant(kv, fp4_block_size, True)`

Hadamard 旋转（`rotate_activation`）先于量化，目的是通过随机正交变换将信息均匀散布到所有维度，使 FP4 量化的损失更均匀，避免局部离群值导致的信息丢失。

------

## 四、总结

| 组件                        | 量化类型           | 存储格式                  | 量化粒度         | 缩放因子格式   |
| :-------------------------- | :----------------- | :------------------------ | :--------------- | :------------- |
| **Indexer Query**           | FP4 模拟 (inplace) | BF16 (Quant-Dequant)      | per-32           | float8_e8m0fnu |
| **Indexer KV cache**        | FP4 模拟 (inplace) | BF16 (Quant-Dequant)      | per-32           | float8_e8m0fnu |
| **主 Attention Activation** | FP8                | float8_e4m3fn             | per-128 (非RoPE) | float8_e8m0fnu |
| **MoE 专家权重**            | FP4                | float4_e2m1fn_x2 (packed) | per-32           | float8_e8m0fnu |
| **MoE 激活值**              | FP8 (动态)         | float8_e4m3fn             | per-128          | float8_e8m0fnu |

**4-bit Indexer** 的核心思想是用一个轻量级（128维）、量化对齐的独立 attention scorer，从压缩 KV 中选 top-k 位置，让稀疏注意力既能覆盖滑动窗口内的精确位置，又能覆盖长距离中被压缩的关键位置。

**MoE FP4 量化** 的核心思路是将专家权重压缩到极致（每个权重只占4bit，packed存储），配合 Power-of-2 Scale（E8M0）避免乘法变除法，再通过专门优化的 TileLang CUDA kernel 做 FP8 act × FP4 weight 的异构 GEMM，在保持推理质量的同时显著降低模型体积和显存带宽需求。