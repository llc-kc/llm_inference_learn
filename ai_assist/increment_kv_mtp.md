## 深入分析：PD Disaggregation Decode Radix Cache 与 Speculative Decoding (MTP/EAGLE) 不兼容的原因

### 1. 背景架构回顾

#### 1.1 PD Disaggregation 的 Decode 工作流程

在 Prefill-Decode 分离架构中：

```
用户请求 → Prefill Server (计算KV) → 网络传输KV → Decode Server (续写token)
```

当 --disaggregation-decode-enable-radix-cache 启用时，Decode Server 会维护自己的 radix tree 缓存：

1. **[Decode 预分配阶段]** ([decode.py:L731-L755](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/decode.py#L731-L755)): 新请求到达时，Decode Server 用 origin_input_ids 在本地 radix cache 中做 prefix matching
2. **[通信 prefix_len]** ([prefill.py:L319-L325](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/prefill.py#L319-L325)): Decode 将匹配到的 decode_prefix_len 告知 Prefill，Prefill 据此跳过已缓存的 prefix，只传输 delta 部分的 KV
3. **[KV 复用]** ([decode.py:L968-L1005](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/decode.py#L968-L1005)): Decode 端直接复用本地 radix cache 中已有的 KV 页面，无需重复传输

#### 1.2 Speculative Decoding (MTP/EAGLE) 的双模型架构

MTP/EAGLE 引入了一个 draft model（草案模型），它与 target model（目标模型）共同工作：

- **Target model**: 主模型，负责最终 token 生成和验证
- **Draft model**: 轻量草案模型，生成候选 token 供 target 验证
- **关键点**: Draft model 有自己独立的 KV cache pool (draft_token_to_kv_pool)，但其 KV 索引与 target model **共享**（indices are shared）

从 [prefill.py:L143-L164](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/prefill.py#L143-L164) 可以看到，Prefill 端初始化 KV manager 时，会将 draft model 的 KV data pointers 与 target 的拼接在一起传输：

```
if self.draft_token_to_kv_pool is not None:
    # We should also transfer draft model kv cache. The indices are
    # always shared with a target model.
    draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
        self.draft_token_to_kv_pool.get_contiguous_buf_infos()
    )
    kv_data_ptrs += draft_kv_data_ptrs  # 拼接传输
    kv_data_lens += draft_kv_data_lens
    kv_item_lens += draft_kv_item_lens
```

------

### 2. 核心不兼容原因：Draft Model KV 未被 Radix Cache 覆盖

#### 2.1 问题根源

这是**最根本的不兼容原因**。Decode 端的 radix cache 只缓存了 **target model 的 KV cache**，但 speculative decoding 需要 **target 和 draft 两个模型的 KV cache 都就位**。

流程分析：

```
时间线 T1: Request A 完成 (origin_input_ids = [tok0, tok1, ..., tok99])
  Prefill: 计算 target KV + draft KV → 传输给 Decode
  Decode:  生成 output tokens → 请求完成
  Radix Cache 插入: cache_finished_req → 只缓存 target KV indices 
                   (draft KV 没有进入 radix cache!)


时间线 T2: Request B 到达 (origin_input_ids = [tok0, tok1, ..., tok99] 前缀相同)
  Decode radix cache: prefix match 命中! prefix_len = 100
  Decode → Prefill: "跳过前 100 个 token，只传后面的"
  
  Prefill 行为: 跳过前 100 个 token 的 KV 传输
    → 跳过了 target KV[0:100]  ✅ Decode 端 radix cache 有
    → 跳过了 draft KV[0:100]  ❌ Decode 端没有!!
  
  Decode 结果: 
    target model 可以正常推理 (KV 从 radix cache 来)
    draft model KV 缺失 → 无法正确生成候选 token → 推理错误
```

#### 2.2 代码证据

**Radix cache 只存储 target KV**，在 [radix_cache.py:L535-L576](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/mem_cache/radix_cache.py#L535-L576) 中：

```
def cache_unfinished_req(self, req: Req, chunked=False):
    token_ids = req.fill_ids
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :len(token_ids)
    ]  # ← 只有 target model 的 KV pool
    
    radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle)
    values = kv_indices[:len(radix_key)]
    result = self.insert(InsertParams(key=radix_key, value=values, ...))
```

**Prefill 传输逻辑**在 [prefill.py:L741-L825](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/prefill.py#L741-L825)，send_kv_chunk 用 start_send_idx = decode_prefix_len 来决定跳过多少 token。KV manager 虽然将 target + draft 的 data pointers 拼接，但传输范围由同一个 start_send_idx 控制，所以 draft KV 也被一并跳过了。

------

### 3. 次要不兼容原因：Bigram Key 语义偏差

当 spec_algorithm.is_eagle() == True 时，radix cache 使用 bigram key 模式（见 [scheduler.py:L848-L851](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/managers/scheduler.py#L848-L851)）：

```
params = CacheInitParams(
    ...
    is_eagle=self.spec_algorithm.is_eagle(),  # → True for EAGLE/MTP
    ...
)
```

Bigram key 的语义是：N 个原始 token → N-1 个 bigram 键。这在 [radix_cache.py:L155-L164](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/mem_cache/radix_cache.py#L155-L164) 中体现：

```
# bigram match logic
matched = max(0, min(i - 1, len(self), len(other)))
# N 个匹配的 raw token → N-1 个匹配的 bigram
```

这导致 prefix_len 在 bigram 模式下比 token 模式下少 1。当这个值传到 Prefill 端作为 decode_prefix_len 时，会与 Prefill 端按 token 计数的语义产生偏差。

虽然这不是最根本的问题（可以通过 +1 修正），但它增加了修复的复杂度。

------

### 4. 第三层不兼容原因：MTP 多 Token 批处理与 KV 生命周期

MTP 每步生成多个 token，其中部分被接受、部分被拒绝。这导致 KV cache 的扩展模式与普通 decode 不同：

- 每个 decode step 可能扩展多个 token 的 KV（而非 1 个）
- 被拒绝的 draft token 的 KV 需要清理
- kv_committed_len 的增长步长 > 1

而在 [decode.py:L1080-L1099](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/decode.py#L1080-L1099) 中，decode radix cache 路径的 _pre_alloc 和 fill_ids 管理假设了简单的单 token decode 语义。虽然代码通过 max(len(output_ids) - 1, 0) 做了一定的兼容，但 MTP 的 accept/reject 模式使得 cache_unfinished_req 的插入和 evict 逻辑可能产生不正确的状态。

------

### 5. 总结

| 层次     | 原因                                                         | 严重程度     |
| -------- | ------------------------------------------------------------ | ------------ |
| **核心** | Decode radix cache 只存 target KV，prefix match 命中后 Prefill 跳过的传输同时丢失了 draft KV，decode 端 draft model 无 KV 可用 | **致命**     |
| **次要** | Bigram key 模式下 prefix_len 比 token 模式少 1，导致 KV 传输范围计算出现 off-by-one | 中等（可修） |
| **工程** | MTP 的多 token batch accept/reject 模式与 decode radix cache 的 insert/evict 生命周期假设不完全兼容 | 中等（可修） |

**最根本的原因**是架构层面的：Radix cache 的设计只考虑了 target model 的单模型 KV 复用场景，而 speculative decoding 引入的 draft model 需要独立的 KV cache，且该 KV 也必须随着 prefix 一起被缓存和复用。当前的实现中，draft KV 不在 radix cache 的管理范围内，因此两者无法共存。



## Prefill 节点为什么可以支持 Speculative Decoding 的 Radix Cache？

首先，需要纠正一个关键前提：**在 PD disaggregation 下，Prefill 节点和 Decode 节点都可以使用 speculation。问题只在 Decode 节点启用 disaggregation-decode-enable-radix-cache 时才会出现。**

所以 Prefill 节点上的 radix cache 与 speculative decoding 是兼容的，原因如下：

### 1. Prefill 节点上 Draft Model 的 KV 是被"重新计算"的，而非从缓存复用

在 Prefill 节点执行一个 extend batch（带 radix cache 命中）时的流程：

**第一步 (EXTEND forward)**：Target model 处理 **delta tokens**（新 token），attention 部分会 attend 到 cached prefix KV + 新计算的 delta KV。关键代码在 [schedule_batch.py:L1667-L1703](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/managers/schedule_batch.py#L1667-L1703)：

```
def prepare_for_extend(self):
    # 只传入 delta tokens!
    input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs]
    seq_lens = [len(r.fill_ids) for r in reqs]     # 全长
    prefix_lens = [len(r.prefix_indices) for r in reqs]  # 缓存的长度
```

Target model 只为 delta 产生 hidden states，但 attention 时 seq_lens 是全长，包含了 prefix 部分。

**第二步 (DRAFT_EXTEND forward)**：Draft model 使用 target 的 hidden states 作为输入，执行全序列的 forward。关键的是，DRAFT_EXTEND 的 seq_lens 是**全长**（prefix + delta），所以 draft model 会为**所有位置**（包括 prefix）计算并写入 KV。

Draft model 的 KV 和 Target model 的 KV **共享相同的 req_to_token_pool 索引**。这一点在 [prefill.py:L155-L164](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/prefill.py#L155-L164) 注释中明确说明：

```
if self.draft_token_to_kv_pool is not None:
    # We should also transfer draft model kv cache. The indices are
    # always shared with a target model.
    draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
        self.draft_token_to_kv_pool.get_contiguous_buf_infos()
    )
    kv_data_ptrs += draft_kv_data_ptrs  # target + draft 的数据指针拼接在一起
```

这意味着：

- req_to_token_pool[req_pool_idx, i] = X → target KV 在 target pool 的索引 X，draft KV 在 draft pool 的索引 X
- **同一个索引 X 在两个 pool 中都有效**

### 2. Prefill 完成时，Draft KV 已经就位

在 Prefill 端的 DRAFT_EXTEND 完成后，draft model 的 KV 已经写入了 draft_token_to_kv_pool 中所有位置（包括 prefix 位置对应的索引）。

然后 KV 传输阶段，Prefill 把 target KV pool 和 draft KV pool 的数据**一并传输**。KV manager 初始化时把两者的 data pointers 拼接：

```
传输数据布局: [target KV data... | draft KV data...]
```

接收端（Decode）也做相同的拼接。所以 Decode 端收到 KV 后，draft model 的所有 KV 数据都在。

### 3. 关键对比：为什么 Decode 端不行？

| 方面                 | Prefill 节点                   | Decode 节点 + radix cache                |
| -------------------- | ------------------------------ | ---------------------------------------- |
| **Target prefix KV** | 从 radix cache 复用            | 从 radix cache 复用                      |
| **Draft prefix KV**  | DRAFT_EXTEND 重新计算写入 pool | **没有任何机制创建！**                   |
| **KV 传输**          | target+draft 全量拼接传输      | Prefill 跳过 prefix 部分 → draft KV 丢失 |

在 Decode 端开启 radix cache 后，Decode 告知 Prefill "跳过 prefix 传输"。Prefill 端 send_kv_chunk 用 start_send_idx = decode_prefix_len 跳过了 prefix 的 KV 传输（[prefill.py:L751-L754](file:///D:/codes/open_engine/sglang/sglang_v0.5.11/python/sglang/srt/disaggregation/prefill.py#L751-L754)）。由于 target 和 draft 的数据指针是拼接在一起的，**跳过前缀传输会同时跳过两者的 KV**。

Decode 端的 radix cache 只缓存了 target KV，**没有 draft KV**。而 Decode 服务器不会执行 DRAFT_EXTEND forward —— 它是纯 decode-only 的，不会重新计算任何 KV。

------

### 4. 图示总结

```
Prefill 节点 (兼容):
═══════════════════════════════════════════════════
  Request A: input=[tok0..tok99]
    ├─ EXTEND: target KV[0:100] 写入 target pool
    ├─ DRAFT_EXTEND: draft KV[0:100] 写入 draft pool  ← 全量写入！
    ├─ cache_finished_req: 只缓存 target KV 到 radix tree
    └─ 传输到 Decode: target KV[0:100] + draft KV[0:100]
    
  Request B: input=[tok0..tok99] (相同前缀)
    ├─ radix cache 命中: prefix_indices = target KV[0:100]
    ├─ EXTEND: 跳过计算 (前缀已缓存), 无 delta tokens
    ├─ DRAFT_EXTEND: draft KV[0:100] 重新写入 draft pool  ← 重新计算！
    └─ 传输到 Decode: draft KV 完整就位 ✅


Decode 节点 + radix cache (不兼容):
═══════════════════════════════════════════════════
  Request A: input=[tok0..tok99]
    ├─ Prefill 传输: target KV[0:100] + draft KV[0:100]
    ├─ Decode cache_finished_req: 只缓存 target KV 到 radix tree
    └─ draft KV 没进入 radix cache
    
  Request B: input=[tok0..tok99] (相同前缀)
    ├─ Decode radix cache 匹配: prefix_len=100
    ├─ 告知 Prefill: "跳过前100个token"
    ├─ Prefill 跳过传输: target KV[0:100] 跳过 + draft KV[0:100] 跳过
    ├─ Decode: target KV 从 radix cache 复用 ✅
    └─ Decode: draft KV 缺失 ❌ → 推理失败!
```

### 5. 总结

Prefill 节点之所以可以同时支持 radix cache 和 speculative decoding，根本原因是：**DRAFT_EXTEND forward pass 每次都会重新计算 draft model 的全序列 KV**（包括 prefix 部分），所以即使 target KV 从 radix cache 复用，draft KV 仍然被新鲜计算并可用。Radix cache 只负责跳过 target 的 KV 计算，不影响 draft model 的 KV 产生。

而 Decode 节点如果开启了 decode radix cache，它会跳过了 prefix 的 KV 传输（包括 draft KV），但 Decode 节点**没有任何机制来产生缺失的 draft KV** —— 它是纯 decode-only 的，从不执行 prefill/draft_extend forward。修复这个问题的难点在于：需要在 Decode 端的 radix cache 中同时管理 draft KV 的缓存，或者让 Decode 端能够在必要时触发 DRAFT_EXTEND 来补充缺失的 draft KV。这两种方案都有显著架构改动成本。





## 解决方案分析
### 方案一：传输时只跳过 Target KV，Draft KV 始终全量传输（推荐）
核心思路 ：保持 Target KV 的 radix cache 传输优化，但 Draft KV 始终从第一个 token 开始传输，不使用 decode_prefix_len 跳过。

具体改动 ：

在 prefill.py:L769-L776 的 send_kv_chunk 中，当前逻辑是：

```
kv_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, start_idx:end_idx
]  # start_idx 已经是 decode_prefix_len，跳过了 prefix
```
需要改为：target KV 用 start_idx=decode_prefix_len ，draft KV 用 start_idx=0 。这需要传输层分离 target 和 draft 的索引范围。

优点 ：

- 改动范围小，只影响 KV transfer 层的索引范围
- Target KV 的 radix cache 优化完全保留
- Draft model 通常只有 1-2 层（如 DeepSeek MTP），额外传输开销小（约 2-5% 的总 KV 数据量）
缺点 ：

- 需要修改传输协议（nixl/mooncake 都需改），让 sender/receiver 知道哪些索引段属于 target，哪些属于 draft
- 当前 kv_data_ptrs 拼接方案假设一个统一的 start_send_idx ，需要拆分成独立的 range
改动量评估 ：中等，集中在 prefill.py 的 send_kv_chunk 和对应的 transfer connector（nixl/mooncake）

### 方案二：Decode 端 Radix Cache 同时缓存 Draft KV
核心思路 ：扩展 radix cache 的 value 结构，让它同时存储 target 和 draft 的 KV 索引，在 prefix match 时同时返回两类索引，在 _pre_alloc 时分别写入两个 pool。

具体改动 ：

1. Cache 存储层 ( radix_cache.py 的 cache_finished_req / cache_unfinished_req )：
   当 speculation 启用时，除了存储 kv_indices_target ，还要存储 kv_indices_draft （通过 draft_token_to_kv_pool 获取）
2. Cache 匹配层 ( _match_prefix_and_lock 在 decode.py:L425-L442 )：
   返回 (prefix_indices_target, prefix_indices_draft, prefix_len) 三元组
3. 预分配层 ( _pre_alloc 在 decode.py:L968-L1099 )：
   
   ```
   # 当前只写 target pool
   self.req_to_token_pool.write(
       (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
   )
   # 需要同时写 draft pool
   self.draft_token_to_kv_pool.write(  
       (req.req_pool_idx, slice(0, prefix_len)), draft_prefix_indices
   )
   ```
4. 内存锁定 ：当 radix tree lock 住 target KV 的 page 时，也需要 lock 住 draft KV pool 中对应索引的 page（防止被 draft allocator 回收）
优点 ：

- 最彻底的优化，target 和 draft KV 都享受缓存命中
- 传输完全消除冗余
缺点 ：

- 改动范围大，需要修改 radix cache 核心数据结构
- Draft KV pool 和 Target KV pool 是 独立的 allocator ，需要在两个 allocator 之间同步 lock
- Radix cache 的 value 目前是单一 tensor，改为 pair 需要改很多地方
- 如果 draft model 参数发生变化，缓存的 draft KV 可能失效
改动量评估 ：大，涉及 radix cache 核心、decode disaggregation、两个 pool allocator

### 方案三：Prefill 端补充计算 Draft KV for Prefix
核心思路 ：当 Prefill 收到 decode_prefix_len > 0 时，在 DRAFT_EXTEND 阶段为 prefix 部分单独计算 draft KV（运行一次 short DRAFT_EXTEND），然后传输给 Decode。

具体改动 ：

在 prefill.py 的 process_batch_result_disagg_prefill 中，在 send_kv_chunk 之前，检测是否有 prefix 跳过，如果有则：

1. 先发送 delta target KV（跳过 prefix，使用 decode_prefix_len）
2. 运行一次轻量 DRAFT_EXTEND for prefix 的 token（输入是 prefix token ids）
3. 发送 draft KV for prefix
优点 ：

- 架构上清晰，Prefill 负责"生产"，Decode 只负责"消费"
- 不需要改动 radix cache 结构
缺点 ：

- 增加 Prefill 端的计算延迟（每请求增加一次 draft forward）
- Prefill 节点需要保留 draft model 的 KV pool 中有用数据直到所有 chunk 发送完
- 如果 decode_prefix_len 较大，增加的计算量不小
改动量评估 ：大，涉及 prefill 调度和 KV 传输的时序改动

### 方案四：混合方案 —— 传输层分离 + 按需 Draft 传输
核心思路 ：在方案一基础上简化。不修改传输协议本身，而是让 Prefill 端在发送 KV 时，对于 draft KV 部分 永远发送全部 。

具体做法：在 send_kv_chunk 被调用时，除了原本的 target KV 发送，额外调用一个 send_draft_kv_chunk ，它使用独立的 start_send_idx=0 。Draft 部分始终从 token 0 开始发送。

实现时利用现有的 req_to_token_pool 共享索引的特点：

```
# target: 从 start_send_idx 开始 (可能是 decode_prefix_len，跳过缓存的)
target_kv_indices = req_to_token_pool[req_pool_idx, start_send_idx:end_idx]

# draft: 始终从 0 开始
draft_kv_indices = req_to_token_pool[req_pool_idx, 0:end_idx]  
# 但只在 start_send_idx > 0 时才需要发送 draft 的 prefix 部分
# delta 部分已经包含在 target 的传输中了
```
优点 ：

- 比方案一更简单，不改变 draft 部分的传输协议
- 可以做成只在 decode_prefix_len > 0 时才额外发送 draft KV prefix
缺点 ：

- 需要 KV transfer engine 支持两次 send 调用（或合并为一次）
- 每次 radix cache 命中仍需传输 draft prefix KV
### 综合推荐
维度 方案一 方案二 方案三 方案四 改动规模 中 大 大 小-中 传输优化程度 好 完美 好 好 架构风险 低 高 中 低 计算开销 无增加 无增加 增加 无增加

推荐方案一 （传输分离），理由：

1. Draft model 通常只有 1-2 层（如 DeepSeek V3 的 MTP 只有 1 个额外 transformer layer），draft KV 占总量比例 < 5%，即使全量传输对 RDMA 带宽影响很小
2. 改动集中在 transfer connector 层，不需要触碰 radix cache 核心数据结构
3. 可以做一个简单的 quick fix：在 send_kv_chunk 中，当 start_send_idx > 0 且 draft_token_to_kv_pool is not None 时，额外对 draft 部分发送 [0:end_idx] 的索引
如果长期来看追求极致性能，方案二是最优选择，但需要较大工程投入。方案一可以作为第一阶段快速上线，后续再演进到方案二。
