# DeepSeek V4 ShadowRadix

本文说明 SGLang 在 DeepSeek V4 中如何用 radix cache 支持复杂 attention。这里的
ShadowRadix 不是一个新的 tree 类型名，而是一套设计方式：

- tree 和 scheduler 只维护一套 full logical index；
- V4 backend 在执行时把 full index 投影到 SWA、C4、C128、C4 indexer 和
  compressor state 等多个物理 pool；
- prefix cache 命中的 req 复用 tree 中保存的 physical full indices，从而自然复用
  这些 indices 对应的 SWA KV、compressed KV 和 compressor state。

关键代码路径：

- `python/sglang/srt/layers/attention/deepseek_v4_backend_radix.py`
- `python/sglang/srt/models/deepseek_v4.py`
- `python/sglang/srt/mem_cache/deepseekv4_memory_pool.py`
- `python/sglang/srt/mem_cache/swa_memory_pool.py`
- `python/sglang/srt/mem_cache/swa_radix_cache.py`
- `python/sglang/srt/mem_cache/compress_state.py`
- `python/sglang/srt/mem_cache/common.py`

## 核心模型

ShadowRadix 的主索引是 full logical loc。它出现在：

- `Req.prefix_indices`
- `req_to_token_pool.req_to_token`
- `SWARadixCache.TreeNode.value`
- `ForwardBatch.out_cache_loc`

对 V4 来说，full logical loc 更像一条稳定的 token 地址线，而不是传统意义上的
full-attention KV cache 地址。V4 没有让 radix tree 直接管理所有 pool 的 key。
tree 只保存 full loc；backend 再通过这些 full loc 找到其他 pool 的地址。

主要投影关系：

```text
full loc
  -> full_to_swa_index_mapping[full loc]
  -> swa loc
  -> swa KV
  -> compressor state slot

full loc // page_size
  -> C4 / C128 / C4 indexer compressed page
```

## Pool 角色

### SWA KV pool

SWA pool 存 V4 attention 主路径使用的 sliding-window KV。V4 backend 中：

- `store_cache()` 把当前新 token 的 KV 写入 SWA pool；
- `get_swa_page_indices()` 从 `req_to_token` 读取最近 `SWA_WINDOW` 个 full loc；
- `full_to_swa_index_mapping` 把这些 full loc 翻译成 SWA loc；
- FlashMLA 使用 SWA loc 读取窗口内 KV。

V4 radix backend 中 `SWA_WINDOW = 128`。当前 paged V4 模式中系统
`page_size = 256`。

### C4 / C128 compressed KV pool

每层的 `compress_ratio` 决定 attention 是否读取 compressed KV：

```text
ratio 0:   只使用 SWA
ratio 4:   使用 SWA + C4 compressed KV + C4 indexer
ratio 128: 使用 SWA + C128 compressed KV
```

C4/C128 compressed KV 是 attention 会读取的最终压缩 KV。它由 compressor 生成，
再写入对应 compressed KV pool。

### C4 indexer pool

C4 indexer 是 C4 attention 的 sidecar。它负责根据 query 和 indexer KV 选择
sparse top-k C4 pages。C4 attention 随后读取这些 page 对应的 C4 compressed KV。

### compressor state pool

compressor state 不是 attention 直接读取的 KV cache。它是 compressor 继续生成
C4/C128/indexer compressed KV 时需要的滚动中间状态。

每个 raw state slot 保存一个 token 的 `kv_score`：

```text
C4 state slot:
  [kv_overlap, kv_current, score_overlap, score_current]

C128 state slot:
  [kv_current, score_current]
```

进入 compressor kernel 前，Python 侧会把 raw state buffer reshape 成：

```text
[-1, compress_ratio, last_dim]
```

因此需要区分两层：

- raw state slot：一个 token 的 compressor 输入状态；
- kernel state block：`compress_ratio` 个 raw slot 组成的一组。

## SWA loc 到 state loc

compressor state 绑定到 SWA physical page，而不是绑定到 req，也不是和 SWA token
一比一同尺寸。

映射公式在 `CompressStatePool.translate_from_swa_loc_to_state_loc()`：

```python
swa_page = swa_loc // swa_page_size
state_loc = swa_page * ring_size + (swa_loc % ring_size)
```

普通模式下：

```text
C4   ring_size = 8
C128 ring_size = 128
```

如果 `swa_page_size = 256`，则一个 SWA page 不是配 256 个 state slots：

```text
每个 SWA page:
  C4 state   只有 8 个 raw slots
  C128 state 只有 128 个 raw slots
```

示例：

```text
C4:
  swa_loc 0   -> state 0
  swa_loc 1   -> state 1
  ...
  swa_loc 7   -> state 7
  swa_loc 8   -> state 0   # 覆盖
  ...
  swa_loc 255 -> state 7

C128:
  swa_loc 0..127   -> state 0..127
  swa_loc 128..255 -> state 0..127  # 覆盖
```

所以 compressor state pool 的大小按 SWA page 数量和 ring size 计算：

```text
c4_state_pool_size   = swa_tokens / swa_page_size * 8
c128_state_pool_size = swa_tokens / swa_page_size * 128
```

这表示 state 只保留当前 SWA page 内 compressor 需要的滚动尾部，而不是为每个
SWA token 永久保存一份 state。

## Prefill 流程

假设 prompt 长度是 `L + E`：

- `L`: prefix cache 命中的 token 数；
- `E`: 当前需要实际计算的 extend token 数。

调度侧只把未命中的 token 送进模型：

```text
input_ids = fill_ids[L:]
seq_len   = L + E
prefix_len = L
```

`alloc_for_extend()` 会分配新 token 的 `out_cache_loc`，然后把完整上下文写入
`req_to_token`：

```text
req_to_token[req, 0:L]     = prefix_indices
req_to_token[req, L:L+E]   = out_cache_loc
```

进入 V4 backend 时，prefix 和 extend 已经是一条连续的 full loc 序列。

backend 再为每个 query token 生成：

- SWA window indices；
- full page table；
- C4/C128 compressed metadata；
- C4 indexer metadata；
- compressor write/load metadata。

## Layer 执行顺序

V4 layer 执行时大致顺序是：

1. 计算当前 extend token 的 q/kv；
2. 写当前 token 的 SWA KV；
3. C4 layer 运行 C4 indexer；
4. C4/C128 layer 运行 core compressor；
5. 执行 attention。

attention 读取：

```text
所有 V4 layer:
  SWA KV

C4 layer:
  SWA KV + C4 compressed KV

C128 layer:
  SWA KV + C128 compressed KV
```

compressor 写入：

```text
C4 layer:
  C4 compressed KV
  C4 attention compress_state
  C4 indexer KV
  C4 indexer compress_state

C128 layer:
  C128 compressed KV
  C128 attention compress_state
```

## C4 overlap 什么时候需要

C4 compressor 每 4 个 token 生成一次 compressed KV，但它的有效压缩窗口是 8 个
token：

```text
前 4 个 token: overlap block
后 4 个 token: current block
```

因此，当要生成新的 C4 compressed KV，且前 4 个 token 不在当前这次
`kv_score_input` 中时，就需要从 C4 compress_state 读取 overlap block。

典型场景：

```text
prefix hit = 256
extend = 256..511

生成 token 259 的 C4:
  需要 252..255 + 256..259
  252..255 来自 prefix，不在当前 extend input 中
  => 需要读取历史 C4 compress_state

生成 token 263 的 C4:
  需要 256..259 + 260..263
  这 8 个 token 都在当前 extend input 中
  => 不需要从 prefix 读取 overlap state
```

decode 更极端：每步只有一个新 token，所以除了第一个 C4 block 的特殊情况，C4
边界上的压缩通常都需要历史 state。

## Prefix cache 如何复用 state

纯 SWA radix tree 场景下，compressor state 不是每个 req 固定分配一块。它跟
SWA physical loc 走。

例如：

1. 第一个 req 计算前缀 `0..511`。
2. allocator 分配 full loc 和 SWA loc，并写入 `full_to_swa_index_mapping`。
3. compressor 根据 full loc 找到 SWA loc，再写入对应 state slots。
4. req finish 后，tree 保存 page-aligned 的 full loc。
5. 第二个 req match 到 `0..511` 时，tree 返回第一条 req 留下的 full loc。
6. 第二个 req 的 `req_to_token[0:512]` 指向这些 full loc。
7. backend 再通过 `full_to_swa_index_mapping` 找到原来的 SWA loc 和 state slots。

也就是说，第二个 req 不是重新分配或拷贝 prefix state，而是复用 tree 中保存的
physical full indices；这些 full indices 仍然指向原来的 SWA KV 和 compressor
state。

## 关键不变量

- radix tree 的 value 是 full loc，不是 SWA loc 或 compressed loc。
- `req_to_token` 是 V4 runtime 的统一上下文视图。
- `full_to_swa_index_mapping` 必须覆盖所有仍可命中的 full loc。
- SWA KV 和 compressor state 的物理生命周期绑定到 SWA loc/SWA page。
- C4/C128 compressed KV 的 page 由 full page 派生。
- C4 overlap state 不能从 C4 compressed KV 反推出；它来自 C4 compressor state。
