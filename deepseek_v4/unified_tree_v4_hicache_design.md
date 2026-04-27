# Unified Tree DeepSeek V4 HiCache 设计

本文档记录 UnifiedRadixTree 为 DeepSeek V4 支持 HiCache 的新设计。当前覆盖：

1. 每个 V4 pool 的 indices 派生机制；
2. HostPool 设计；
3. HybridAssembler 设计；
4. HybridController 设计。

后续再补充 Tree Component 和测试方案。

## 设计原则

V4 HiCache 仍然以 `FULL` 作为 radix tree 的 logical skeleton：

```text
FULL node value
  -> prefix cache 的主索引
  -> HiCache anchor
  -> C4 / C128 / C4 indexer 的 page 派生来源

SWA component value
  -> suffix executable window
  -> SWA KV 的 transfer indices
  -> compress_state 的 page 派生来源
```

因此不能把所有 V4 状态都从 full indices 派生。V4 pool 分成两类：

```text
FULL-derived:
  C4 compressed KV
  C4 indexer KV
  C128 compressed KV

SWA-derived:
  SWA KV
  C4 attention compress_state
  C4 indexer compress_state
  C128 attention compress_state
```

## 1. Pool Indices 派生机制

### 1.1 FULL logical anchor

`FULL` 是 tree 的主干。node 上保存的是 full logical/device indices：

```text
node.component_data[FULL].value
```

Host 侧对应 `LogicalHostPool` 中的 logical indices：

```text
node.component_data[FULL].host_value
```

V4 下 `FULL` host pool 不保存真实 Full KV tensor。它只提供 page-aligned logical slots，
用于维持 UnifiedRadixTree 的 host-backed 状态机：

```text
node.backuped
node.evicted
host_hit_length
host leaf eviction
```

约束：

```text
FULL host allocation 必须按 full_page_size 对齐
```

否则后续 `full_indices // full_page_size` 派生出的 compressed page indices 会不稳定。

### 1.2 C4 compressed KV

C4 compressed KV 从 FULL page 派生：

```text
c4_page_indices = unique(full_indices // full_page_size)
```

当前 V4 中：

```text
full_page_size = 256
c4_page_size = full_page_size // 4 = 64
```

一个 full page 的 256 个 token 正好对应一个 C4 compressed page 的 64 个 compressed
entries，因此 C4 HiCache 以 full page 为派生单位。

### 1.3 C4 indexer KV

C4 indexer KV 与 C4 compressed KV 使用同一组 page indices：

```text
c4_indexer_page_indices = unique(full_indices // full_page_size)
```

它的 tensor layout 和 item bytes 与 C4 compressed KV 不同，但 page identity 一样：

```text
one full page -> one C4 indexer page
```

### 1.4 C128 compressed KV

C128 compressed KV 同样从 FULL page 派生：

```text
c128_page_indices = unique(full_indices // full_page_size)
```

当前 V4 中：

```text
full_page_size = 256
c128_page_size = full_page_size // 128 = 2
```

一个 full page 的 256 个 token 正好对应一个 C128 compressed page 的 2 个 compressed
entries。

### 1.5 SWA KV

SWA KV 不能从 full page 直接派生。它来自 `SWAComponent.value`：

```text
swa_indices = node.component_data[SWA].value
```

`SWAComponent` 在 insert 时已经根据 `swa_evicted_seqlen` 维护 suffix window：

```text
window 外:
  SWA value = None

window 内:
  SWA value = full_to_swa_index_mapping[full_value]

跨 boundary:
  split node
  parent: SWA tombstone
  child: SWA value
```

所以 HiCache backup 时只要 `SWAComponent.value is not None`，就说明这个 node 上的
SWA KV 仍属于可执行 suffix window，应该参与 backup。

loadback 时从 leaf 往 parent 收集 `SWA.host_value`，直到覆盖 `sliding_window_size`
或没有更多 host SWA data。device 侧通过 `alloc_full_with_suffix_swa(full_len,
swa_suffix_len)` 分配：

```text
full device indices: 整段 host hit prefix
swa device indices: suffix window
```

并重建：

```text
full_to_swa_index_mapping[new_full_suffix] = new_swa_indices
```

### 1.6 C4 attention compress_state

C4 attention state 从 SWA indices 派生，而不是从 FULL page 派生。

V4 state 地址关系是：

```text
full loc -> swa loc -> state loc
```

其中：

```python
swa_page = swa_loc // swa_page_size
state_loc = swa_page * c4_ring_size + (swa_loc % c4_ring_size)
```

普通模式下：

```text
c4_ring_size = 8
```

HostPool transfer 的外部 indices 使用 SWA token indices：

```text
state_transfer_indices = swa_indices
```

HostPool 内部再转换成 state page rows：

```text
state_rows = unique(swa_indices // swa_page_size)
```

每个 state row 表示一个 SWA page 对应的一整段 C4 state ring：

```text
row bytes = c4_ring_size * raw_c4_state_slot_bytes
```

### 1.7 C4 indexer compress_state

C4 indexer state 与 C4 attention state 使用相同的 indices 语义：

```text
state_transfer_indices = swa_indices
state_rows = unique(swa_indices // swa_page_size)
```

区别在于 item bytes 和 layer mapping：

```text
head_dim = indexer_head_dim
layers = ratio == 4 的 layers
```

### 1.8 C128 attention compress_state

C128 state 也从 SWA indices 派生：

```text
state_transfer_indices = swa_indices
state_rows = unique(swa_indices // swa_page_size)
```

普通模式下：

```text
c128_ring_size = 128
row bytes = c128_ring_size * raw_c128_state_slot_bytes
```

在当前 `full_page_size=256` 的严格 page-aligned loadback 中，C128 state 理论上不是
所有场景都必需，因为 full page boundary 也是 C128 boundary。但为了保持统一恢复语义，
并兼容 chunked/partial loadback 和未来 page size 变化，设计上保留 C128 state
offload。

### 1.9 ratio == 0 的 layer

`compression_ratio == 0` 的 layer 只有 SWA：

```text
需要:
  SWA KV

不需要:
  C4 compressed KV
  C4 indexer KV
  C128 compressed KV
  compress_state
```

Assembler 构建 layer mapping 时必须跳过 ratio 0 layer。HostPool 可以存在，但对应
PoolEntry 的 layer mapper 应该对 ratio 0 返回 `None`。

## 2. HostPool 设计

### 2.1 总体拓扑

采用多个 HostPool，贴合现有 `HostPoolGroup` / `PoolEntry` 模型：

```text
LogicalHostPool
  FULL logical anchor

DeepSeekV4PagedHostPool
  SWA KV

DeepSeekV4PagedHostPool
  C4 compressed KV

DeepSeekV4PagedHostPool
  C4 indexer KV

DeepSeekV4PagedHostPool
  C128 compressed KV

DeepSeekV4StateHostPool
  C4 attention compress_state

DeepSeekV4StateHostPool
  C4 indexer compress_state

DeepSeekV4StateHostPool
  C128 attention compress_state
```

不做大的 `DeepSeekV4TokenToKVPoolHost` owner。原因是现有 controller 已经按
`PoolEntry` 分发 transfer，多 HostPool 能保持语义清楚，也减少改动面。

第一版保留两个 V4 专用 HostPool：

```text
DeepSeekV4PagedHostPool:
  mirror 真实 paged tensor list
  用于 SWA KV、C4 KV、C4 indexer KV、C128 KV

DeepSeekV4StateHostPool:
  mirror per-layer CompressStatePool list
  用于 C4 state、C4 indexer state、C128 state
```

两者都需要实现 `io_backend` 分支。当前先支持 `direct`，但这里的 `direct` 不是
`torch.Tensor.copy_`，而是现有 HiCache 使用的 `transfer_kv_direct` API；其他 backend
先 fail fast，避免静默走错 copy 语义。

### 2.2 LogicalHostPool

`LogicalHostPool` 是 `KV` anchor pool。它只管理 logical host indices：

```text
size = num_host_pages * full_page_size
page_size = full_page_size
```

要求：

```text
alloc/free 必须以 full page 为稳定单位
```

即使外部请求 token span，内部也应该返回 page-aligned logical spans，避免 host
fragmentation 导致 compressed page 派生不稳定。

### 2.3 Paged KV HostPool

SWA、C4、C4 indexer、C128 都使用 `DeepSeekV4PagedHostPool`。

每个 pool 的 host tensor layout：

```text
per local layer:
  host_buffer[layer] = [num_host_pages, item_bytes]
```

其中：

```text
C4 item_bytes:
  c4_kv_pool.bytes_per_page_padded

C128 item_bytes:
  c128_kv_pool.bytes_per_page_padded

C4 indexer item_bytes:
  c4_indexer_pool.page_size * indexer_head_dim
  + c4_indexer_pool.page_size * num_scales_per_token * 4

SWA item_bytes:
  swa_kv_pool.bytes_per_page_padded
```

C4、C4 indexer、C128 不需要独立 allocator。它们的 host/device indices 都从 FULL
anchor 派生：

```text
host_page_indices   = unique(full_host_indices // full_page_size)
device_page_indices = unique(full_device_indices // full_page_size)
```

SWA 是独立 transfer。它的 host/device indices 来自 SWA allocator 或 SWA host pool：

```text
host_rows = unique(swa_host_indices // swa_page_size)
device_rows = unique(swa_device_indices // swa_page_size)
```

### 2.4 SWA HostPool

SWA HostPool 保存 suffix window 的 SWA KV。

外部 transfer 使用 SWA token indices：

```text
host_indices = swa_host_indices
device_indices = swa_device_indices
```

HostPool 内部按 SWA page 搬：

```text
host_rows = unique(swa_host_indices // swa_page_size)
device_rows = unique(swa_device_indices // swa_page_size)
```

每个 row 对应一个完整 SWA page 的 KV。

### 2.5 State HostPool

C4 state、C4 indexer state、C128 state 使用独立的 `DeepSeekV4StateHostPool`。
一共创建三个实例：

```text
c4_state_host_pool:
  C4 attention CompressStatePool

c4_indexer_state_host_pool:
  C4 indexer CompressStatePool

c128_state_host_pool:
  C128 attention CompressStatePool
```

它们也保持三个 PoolName / PoolEntry：

外部协议上不要合并成一个 `DSV4_STATE` PoolName，原因是三者：

- layer mapping 不同；
- item bytes 不同；
- device buffer list 不同；
- ratio 0/4/128 的存在条件不同。

V4 原生 state 的 device 结构是 per-global-layer pool list：

```text
kvcache.compress_state_pools[global_layer_id]
  -> CompressStatePool | None

kvcache.indexer_compress_state_pools[global_layer_id]
  -> CompressStatePool | None
```

单个 `CompressStatePool` 内部没有 layer 维度，只有 state slots/pages：

```text
pool.kv_score_buffer.kv_score[state_loc]
```

所以 `DeepSeekV4StateHostPool` 不接收普通 device KV pool，而是接收已经筛好的
per-layer `CompressStatePool` list：

```python
DeepSeekV4StateHostPool(
    pool_name="c4_state",
    state_pools=[
        kvcache.compress_state_pools[global_layer_id]
        for global_layer_id in c4_state_global_layers
    ],
    num_host_pages=num_host_pages,
    swa_page_size=kvcache.swa_page_size,
)
```

它内部 host tensor layout：

```text
per selected layer:
  host_buffer[layer] = [num_host_pages, state_page_bytes]
```

其中：

```text
state_slot_bytes = state_pool.kv_score_buffer.kv_score[0].nbytes
state_page_bytes = state_pool.ring_size * state_slot_bytes
```

state HostPool 不独立分配 indices。它们共享 SWA transfer 的 host/device indices：

```text
PoolTransfer(
    name=PoolName.DEEPSEEK_V4_C4_STATE,
    device_indices_source=PoolName.SWA,
)
```

Controller resolve 后：

```text
C4_STATE.host_indices = SWA.host_indices
C4_STATE.device_indices = SWA.device_indices
```

然后 state HostPool 内部做：

```text
host_rows = unique(swa_host_indices // swa_page_size)
device_rows = unique(swa_device_indices // swa_page_size)
```

每个 row 是一个 SWA page 对应的 state ring：

```text
C4 state row:
  c4_ring_size * raw_c4_state_slot_bytes

C4 indexer state row:
  c4_ring_size * raw_c4_indexer_state_slot_bytes

C128 state row:
  c128_ring_size * raw_c128_state_slot_bytes
```

`DeepSeekV4StateHostPool` 内部需要把 device state buffer 视为 page rows：

```text
device_state_rows = state_pool.kv_score_buffer.kv_score
  .view(uint8)
  .reshape(num_state_rows, state_page_bytes)
```

其中 `num_state_rows` 至少覆盖所有可能的 `swa_loc // swa_page_size`。

因此 `DeepSeekV4StateHostPool` 内部有三组并行结构：

```text
state_pools[local_layer_id]
  -> selected global layer 的 CompressStatePool

device_page_views[local_layer_id]
  -> state_pools[local_layer_id] reshape 出来的 page-row view

host_buffers[local_layer_id]
  -> 对应 host page-row buffer
```

`load_to_device_per_layer()` 时，`HostPoolGroup` 已经用 `PoolEntry.layer_mapper` 把
global layer id 翻译成 state HostPool 的 `local_layer_id`。StateHostPool 只需要用
这个 local id 定位 host/device page-row：

```text
local_layer_id
  -> host_buffers[local_layer_id]
  -> device_page_views[local_layer_id]
  -> transfer page rows
```

它不需要再理解该 layer 是 C4、C4 indexer 还是 C128；这个语义已经由三个不同的
StateHostPool 实例和 PoolEntry 隔离开。

### 2.6 D→H 传输

Backup 时，`HostPoolGroup.backup_from_device_all_layer()` 遍历 `PoolTransfer` list。
每个 transfer 根据 `PoolEntry.name` 找到对应 HostPool：

```text
FULL anchor:
  LogicalHostPool 只分配 host indices，不做真实 tensor copy

C4 / C4 indexer / C128:
  full-derived page rows

SWA:
  swa-derived page rows

state:
  share SWA indices, then derive state rows
```

对于 V4 state，D→H 搬的是 device state buffer 中一个 SWA page 对应的 ring block。
第一版 `io_backend == "direct"` 时通过 `transfer_kv_direct` 搬 page-row；其他 backend
先报错。

### 2.7 H→D 传输

Loadback 时，`HostPoolGroup.load_to_device_per_layer()` 对同一个 global layer 可以执行
多个 PoolTransfer：

```text
ratio == 0:
  SWA

ratio == 4:
  SWA
  C4
  C4 indexer
  C4 attention state
  C4 indexer state

ratio == 128:
  SWA
  C128
  C128 attention state
```

每个 PoolEntry 的 `layer_mapper(global_layer_id)` 决定当前 global layer 是否需要搬：

```text
SWA mapper:
  all layers

C4 / C4 indexer / C4 state mapper:
  ratio == 4 layers

C128 / C128 state mapper:
  ratio == 128 layers

ratio == 0:
  compressed/state mapper 返回 None
```

同一 layer 的所有 transfer 在同一 load stream 中完成后，layer done event 才能标记该
layer ready。
第一版 `io_backend == "direct"` 时通过 `transfer_kv_direct` 搬当前 layer 的
page-row；其他 backend 先报错。

### 2.8 HostPool 必须提供的接口

每个 V4 HostPool 需要兼容现有 `HostPoolGroup` 调用：

```python
backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)

load_to_device_per_layer(
    device_pool,
    host_indices,
    device_indices,
    layer_id,
    io_backend,
)

get_page_buffer_meta(indices)
get_dummy_flat_data_page()
set_from_flat_data_page(index, data_page)
```

其中 `get_page_buffer_meta()` 用于 storage backend 注册和 zero-copy I/O。对于
`slot_page_size` 不为空的 HostPool，必须先把 token indices 转成 page rows：

```text
rows = unique(indices // slot_page_size)
```

再返回每个 row、每个 local layer 的 host pointer。

### 2.9 IO backend 支持

`DeepSeekV4PagedHostPool` 和 `DeepSeekV4StateHostPool` 都要显式处理 `io_backend`。
第一版只启用：

```text
io_backend == "direct"
```

`direct` 的语义是复用 `sgl_kernel.kvcacheio.transfer_kv_direct`：

```python
D→H:
  transfer_kv_direct(
      src_layers=device_layer_buffers,
      dst_layers=host_layer_buffers,
      src_indices=device_page_rows,
      dst_indices=host_page_rows,
      page_size=1,
  )

H→D:
  transfer_kv_direct(
      src_layers=[host_layer_buffer],
      dst_layers=[device_layer_buffer],
      src_indices=host_page_rows,
      dst_indices=device_page_rows,
      page_size=1,
  )
```

这里 `page_size=1` 是因为 V4 HostPool 已经把外部 token indices 转成了 page-row
indices。每个 row 自身就是一个完整 blob：

```text
Paged KV:
  row bytes = compressed/swa KV page bytes

State:
  row bytes = one SWA page's state ring bytes
```

对 state pool，`device_buffer[local_layer]` 是从 `CompressStatePool` reshape 出来的
state page view；copy 的单位是一整个 state ring page。

不支持的 backend 必须 fail fast：

```text
raise NotImplementedError(
    f"{pool_name} supports only direct io_backend, got {io_backend}"
)
```

后续补 `kernel` 时也应该走现有 kvcacheio API，而不是普通 tensor copy：

```text
H→D one layer:
  transfer_kv_per_layer_mla(..., item_size=row_bytes)

D→H all layers:
  transfer_kv_all_layer_mla(..., item_size=row_bytes)
```

storage zero-copy 仍通过 `get_page_buffer_meta()` 暴露 host page-row pointer。

## 3. HybridAssembler 设计

Assembler 只负责把 V4 device pool 拆成 `PoolEntry` 拓扑，不参与每次 backup/loadback
的动态决策。

### 3.1 输入约束

V4 HiCache stack 需要同时看到三个 component：

```text
FULL
SWA
DSV4_COMPRESSED
```

其中 `FULL` 是主 anchor，`SWA` 提供 suffix window，`DSV4_COMPRESSED` 负责生成
C4/C128/indexer/state 的 PoolTransfer。

device 侧需要：

```text
DeepSeekV4TokenToKVPool
SWATokenToKVPoolAllocator
```

不能用普通 full allocator 代替 SWA allocator。loadback 时必须通过 SWA allocator
建立：

```text
full loc -> swa loc
```

否则 attention backend 无法通过 `full_to_swa_index_mapping` 找到 SWA KV 和
compress_state。

### 3.2 Layer mapping

`PoolEntry.layer_mapper` 解决的是 global layer id 到当前 `PoolEntry` 对应 HostPool
内部 layer/buffer 下标的翻译问题。

V4 的 layer 在模型里有统一的 global layer id：

```text
global layer id = model layer id
```

但不同物理状态的组织方式不一样：

```text
C4 KV pool:
  index is compressed local layer id
  only stores ratio == 4 layers

C4 indexer KV pool:
  index is compressed local layer id
  only stores ratio == 4 layers

C128 KV pool:
  index is compressed local layer id
  only stores ratio == 128 layers

SWA KV pool:
  index is global layer id
  stores all layers

CompressStatePool:
  one pool object per global layer
  no layer dimension inside one CompressStatePool
```

因此 Assembler 需要构建两类 mapping。

第一类是复用 V4 原生 `compress_layer_id` 的 mapping：

```text
C4 / C4 indexer:
  ratio == 4 layers -> compact local layer id

C128:
  ratio == 128 layers -> compact local layer id
```

第二类是 state HostPool 自己的 selected-layer list mapping：

```text
C4 state / C4 indexer state:
  ratio == 4 layers -> index in DeepSeekV4StateHostPool.state_pools

C128 state:
  ratio == 128 layers -> index in DeepSeekV4StateHostPool.state_pools

SWA:
  global layer id -> same id

ratio == 0:
  only SWA mapper returns a layer id
```

例如有 5 个 global layers：

```text
global layer:       0  1  2  3  4
compression ratio:  0  4  4 128 0
```

各 pool 内部实际保存的是：

```text
SWA KV pool:
  global 0 -> local 0
  global 1 -> local 1
  global 2 -> local 2
  global 3 -> local 3
  global 4 -> local 4

C4 KV / C4 indexer KV pool:
  global 1 -> local 0
  global 2 -> local 1

C128 KV pool:
  global 3 -> local 0

C4 state / C4 indexer state HostPool state_pools:
  global 1 -> list index 0
  global 2 -> list index 1

C128 state HostPool state_pools:
  global 3 -> list index 0
```

所以 mapper 行为是：

```text
swa_layer_mapper(0)  -> 0
swa_layer_mapper(1)  -> 1
swa_layer_mapper(2)  -> 2
swa_layer_mapper(3)  -> 3
swa_layer_mapper(4)  -> 4

c4_layer_mapper(0)   -> None
c4_layer_mapper(1)   -> 0
c4_layer_mapper(2)   -> 1
c4_layer_mapper(3)   -> None
c4_layer_mapper(4)   -> None

c128_layer_mapper(0) -> None
c128_layer_mapper(1) -> None
c128_layer_mapper(2) -> None
c128_layer_mapper(3) -> 0
c128_layer_mapper(4) -> None
```

`None` 的含义是：当前 global layer 不属于这个 PoolEntry，`HostPoolGroup` 在这一层
不需要对该 pool 发起 copy。

这里尤其要区分 state：

```text
kvcache.compress_state_pools[global_layer_id]
  -> returns a per-layer CompressStatePool
  -> that object only has state slots/pages, no layer axis
```

`DeepSeekV4StateHostPool` 会把需要 offload 的 per-layer `CompressStatePool` 收集成
一个 `state_pools` list。`layer_mapper` 返回的是这个 list 的下标，不是
`CompressStatePool` 内部的 layer id。

例如：

```text
c4_state_host_pool.state_pools[0]
  -> kvcache.compress_state_pools[global layer 1]

c4_state_host_pool.state_pools[1]
  -> kvcache.compress_state_pools[global layer 2]

c4_indexer_state_host_pool.state_pools[0]
  -> kvcache.indexer_compress_state_pools[global layer 1]

c128_state_host_pool.state_pools[0]
  -> kvcache.compress_state_pools[global layer 3]
```

Assembler 构建 state mapping 时应同时保存 global layer list：

```python
c4_state_global_layers = []
c128_state_global_layers = []
c4_state_mapping = {}
c128_state_mapping = {}

for global_layer_id, ratio in enumerate(kvcache.compression_ratios):
    if ratio == 4:
        c4_state_mapping[global_layer_id] = len(c4_state_global_layers)
        c4_state_global_layers.append(global_layer_id)
    elif ratio == 128:
        c128_state_mapping[global_layer_id] = len(c128_state_global_layers)
        c128_state_global_layers.append(global_layer_id)
```

然后用这些 global layer list 构建 `DeepSeekV4StateHostPool`：

```python
c4_state_host_pool = DeepSeekV4StateHostPool(
    pool_name="c4_state",
    state_pools=[
        kvcache.compress_state_pools[layer_id]
        for layer_id in c4_state_global_layers
    ],
    num_host_pages=num_host_pages,
    swa_page_size=kvcache.swa_page_size,
)

c4_indexer_state_host_pool = DeepSeekV4StateHostPool(
    pool_name="c4_indexer_state",
    state_pools=[
        kvcache.indexer_compress_state_pools[layer_id]
        for layer_id in c4_state_global_layers
    ],
    num_host_pages=num_host_pages,
    swa_page_size=kvcache.swa_page_size,
)

c128_state_host_pool = DeepSeekV4StateHostPool(
    pool_name="c128_state",
    state_pools=[
        kvcache.compress_state_pools[layer_id]
        for layer_id in c128_state_global_layers
    ],
    num_host_pages=num_host_pages,
    swa_page_size=kvcache.swa_page_size,
)
```

`DeepSeekV4StateHostPool` 初始化时要验证 `state_pools` 中没有 `None`，并且同一个
state HostPool 内的 `ring_size`、slot bytes 一致。

### 3.3 PoolEntry 拓扑

推荐 entry 拓扑：

```text
KV
  pool: LogicalHostPool
  role: primary index anchor

SWA
  pool: SWA KV HostPool
  source: independent SWA transfer
  mapper: all layers

DEEPSEEK_V4_C4
  pool: C4 compressed KV HostPool
  indices: derive from KV full page
  mapper: ratio == 4

DEEPSEEK_V4_C4_INDEXER
  pool: C4 indexer KV HostPool
  indices: derive from KV full page
  mapper: ratio == 4

DEEPSEEK_V4_C128
  pool: C128 compressed KV HostPool
  indices: derive from KV full page
  mapper: ratio == 128

DEEPSEEK_V4_C4_STATE
  pool: DeepSeekV4StateHostPool(C4 attention state)
  indices: source = SWA
  mapper: ratio == 4

DEEPSEEK_V4_INDEXER_STATE
  pool: DeepSeekV4StateHostPool(C4 indexer state)
  indices: source = SWA
  mapper: ratio == 4

DEEPSEEK_V4_C128_STATE
  pool: DeepSeekV4StateHostPool(C128 attention state)
  indices: source = SWA
  mapper: ratio == 128
```

FULL-derived entries 使用同一个 derive 函数：

```python
def derive_full_page_indices(full_indices):
    return torch.unique(full_indices.to(torch.int64) // full_page_size)
```

state entries 不写 derive 函数。它们通过 `device_indices_source=PoolName.SWA`
复用 SWA transfer 的 token indices，由 `DeepSeekV4StateHostPool` 内部转 state page
rows。

### 3.4 ratio == 0 兼容

某些 layer 的 `compression_ratio` 可能是 0。这种 layer 没有 compressed KV，也没有
compress_state，只需要 SWA。

Assembler 应按实际 mapping 创建 pool：

```text
has_c4 = any(ratio == 4)
has_c128 = any(ratio == 128)
```

当 `has_c4 == False` 时，不创建：

```text
DEEPSEEK_V4_C4
DEEPSEEK_V4_C4_INDEXER
DEEPSEEK_V4_C4_STATE
DEEPSEEK_V4_INDEXER_STATE
```

当 `has_c128 == False` 时，不创建：

```text
DEEPSEEK_V4_C128
DEEPSEEK_V4_C128_STATE
```

Component 也应只 emit 已存在 mapping 对应的 transfer。这样 Controller 遇到未知 pool
就可以视为真实错误，而不是静默跳过。

### 3.5 Entry 顺序

Assembler 建议输出稳定顺序：

```text
KV
SWA
C4
C4 indexer
C128
C4 state
C4 indexer state
C128 state
```

这个顺序便于 debug，也让日志中先看到 anchor/source pool，再看到依赖它们的 pool。

Unified tree 侧 V4 component 初始化顺序是：

```text
FULL
SWA
DSV4_COMPRESSED
```

所以 component 生成 transfer 时，SWA transfer 通常在 DSV4 compressed/state transfer
之前出现。Controller 可以利用这个事实保持实现简单，但不能在 source 缺失或 source
indices 为空时静默跳过。

### 3.6 构建失败策略

Assembler 应在初始化阶段 fail fast：

```text
存在 ratio == 4:
  必须能找到 C4 KV、C4 indexer KV、C4 state、indexer state 的 device buffer

存在 ratio == 128:
  必须能找到 C128 KV、C128 state 的 device buffer

存在任意 layer:
  必须能找到 SWA device pool 和 SWA allocator
```

如果某类 compressed ratio 不存在，则对应 pool 可以完全不创建。

## 4. HybridController 设计

Controller 的核心职责是把一个 `CacheOperation` 中的多路 `PoolTransfer` 解析成可执行的
host/device indices，然后交给 `HostPoolGroup` 做实际 I/O。现有主流程已经能覆盖 V4
需要的大部分能力。

### 4.1 已满足的现有能力

现有 `extra_pools: list[PoolTransfer]` 接口可以表达 V4 需要的多路 transfer：

```text
SWA
C4
C4 indexer
C128
C4 state
C4 indexer state
C128 state
```

`HostPoolGroup.load_to_device_per_layer()` 已经按 layer 遍历所有 `pool_transfers`，
所以同一个 global layer 同时搬 SWA + compressed + state 不需要改 Controller 主流程。

```text
ratio == 4:
  SWA + C4 + C4 indexer + C4 state + C4 indexer state

ratio == 128:
  SWA + C128 + C128 state

ratio == 0:
  SWA
```

`load()` / `write()` 的对外接口也不需要改，继续通过 `extra_pools` 传递 V4 sidecar
transfer。

### 4.2 SWA loadback 分配

loadback 时只要 operation 里有 SWA transfer，就必须继续走 SWA allocator：

```python
full_device_indices, swa_device_indices = alloc_full_with_suffix_swa(
    len(kv_host_indices),
    swa_xfer.swa_suffix_tokens,
)
```

这一步同时完成：

```text
full device slots allocation
swa suffix slots allocation
full_to_swa_index_mapping rebuild
```

需要补一个校验：

```text
len(swa_xfer.host_indices) == swa_xfer.swa_suffix_tokens
```

否则 SWA copy 的 token 数和 allocator 建立的 full->swa 映射长度不一致。

如果没有 SWA transfer，Controller 才能退回普通 full allocator。

### 4.3 不需要改的部分

这次不要扩大 Controller 改动：

- 不改 `load()` / `write()` 对外接口；
- 不让 Controller 判断某个 layer 是 C4 还是 C128；
- 不让 Controller 计算 state loc；
- 不让 Controller 读取或修改 `full_to_swa_index_mapping`；
- 不让 Controller 为 state pool 独立分配 token/page indices；
- 不改 HostPoolGroup 的 per-layer 多 transfer 主流程。

这些逻辑分别属于：

```text
layer ratio -> PoolEntry.layer_mapper
state loc/page row -> DeepSeekV4StateHostPool
full_to_swa_index_mapping -> SWA allocator
emit transfer -> DSV4 component
```

## 5. DeepSeekV4CompressedComponent 设计

V4 特殊 transfer 逻辑统一放在 `DeepSeekV4CompressedComponent`。它不作为一个独立
cache pool 参与 match、lock、evict，也不在 node 上保存 device value。

它只做一件事：

```text
根据 FULL / SWA component 已有状态，生成 V4 sidecar PoolTransfer。
```

### 5.1 Component 边界

`DeepSeekV4CompressedComponent` 不负责：

- 分配 host/device indices；
- 计算 state loc；
- 更新 `full_to_swa_index_mapping`；
- 维护独立 LRU；
- 决定某个 global layer 是 C4 还是 C128。

这些分别属于：

```text
indices allocation -> HybridController / PoolEntry
state page row -> DeepSeekV4StateHostPool
full_to_swa_index_mapping -> SWA allocator
layer mapping -> PoolEntry.layer_mapper
```

组件初始化顺序是：

```text
FULL
SWA
DSV4_COMPRESSED
```

因此 `DeepSeekV4CompressedComponent` 可以读取：

```text
node.component_data[FULL].value / host_value
node.component_data[SWA].value / host_value
```

但它不直接修改 FULL 或 SWA component data。

### 5.2 可用 Pool 判断

ratio == 0 时不存在 compressed/state pool，所以 component 不能无条件 emit 所有
PoolTransfer。它应根据 Assembler 注册的 `PoolEntry` 判断哪些 pool 可用。

建议提供一个小 helper：

```python
def has_pool(self, name: PoolName) -> bool:
    controller = self.cache.cache_controller
    if controller is None:
        return False
    return name in controller.mem_pool_host.entry_map
```

只有 `has_pool(name)` 为真时才 emit 对应 transfer：

```text
has C4 mapping:
  emit DEEPSEEK_V4_C4
  emit DEEPSEEK_V4_INDEXER
  emit DEEPSEEK_V4_C4_STATE
  emit DEEPSEEK_V4_INDEXER_STATE

has C128 mapping:
  emit DEEPSEEK_V4_C128
  emit DEEPSEEK_V4_C128_STATE
```

这样 ratio==0 layer 由 Assembler/PoolEntry 和 component emit 共同规避，不需要
Controller 做特殊判断。

### 5.3 BACKUP_HOST transfer

Backup 时，FULL-derived compressed KV 从 FULL device indices 派生 page indices：

```python
full_device = node.component_data[FULL].value
page_device = unique(full_device // full_page_size)
```

生成：

```text
DEEPSEEK_V4_C4:
  device_indices = page_device

DEEPSEEK_V4_INDEXER:
  device_indices = page_device

DEEPSEEK_V4_C128:
  device_indices = page_device
```

host indices 不在 component 内计算，由 `PoolEntry.derive_indices_fn` 根据 FULL
host anchor 派生。

State transfer 从 SWA device indices 关联：

```python
swa_device = node.component_data[SWA].value
```

如果 `swa_device is None`，说明该 node 不在 suffix executable window 内，不 emit
state transfer。

否则生成：

```text
DEEPSEEK_V4_C4_STATE:
  device_indices_source = SWA

DEEPSEEK_V4_INDEXER_STATE:
  device_indices_source = SWA

DEEPSEEK_V4_C128_STATE:
  device_indices_source = SWA
```

state transfer 不独立分配 host/device indices。Controller 处理 SWA transfer 后，
这些 state transfer 复用：

```text
state.host_indices = swa.host_indices
state.device_indices = swa.device_indices
```

`DeepSeekV4StateHostPool` 再把 SWA token indices 转成 state page rows。

### 5.4 LOAD_BACK transfer

Loadback 时，FULL-derived compressed KV 从 FULL host chain 派生：

```text
walk evicted chain from leaf to parent
collect node.component_data[FULL].host_value
reverse to prefix order
full_host = cat(parts)
page_host = unique(full_host // full_page_size)
```

生成：

```text
DEEPSEEK_V4_C4:
  host_indices = page_host

DEEPSEEK_V4_INDEXER:
  host_indices = page_host

DEEPSEEK_V4_C128:
  host_indices = page_host
```

device indices 不在 component 内计算，由 `PoolEntry.derive_indices_fn` 根据 FULL
device indices 派生。

State loadback 需要跟随 SWA suffix window，而不是 FULL host chain。component 应收集
和 `SWAComponent` 相同的 suffix host span：

```text
walk evicted chain from leaf to parent
collect node.component_data[SWA].host_value
stop when collected tokens >= sliding_window_size
reverse to prefix order
swa_host = cat(parts)
```

如果没有 SWA host data，不 emit state transfer。

否则生成：

```text
DEEPSEEK_V4_C4_STATE:
  host_indices = swa_host
  device_indices_source = SWA

DEEPSEEK_V4_INDEXER_STATE:
  host_indices = swa_host
  device_indices_source = SWA

DEEPSEEK_V4_C128_STATE:
  host_indices = swa_host
  device_indices_source = SWA
```

SWA device indices 由 `alloc_full_with_suffix_swa()` 分配并写入 SWA transfer。
state transfer 通过 `device_indices_source=PoolName.SWA` 复用这段 SWA device indices。

### 5.5 commit_hicache_transfer

`DeepSeekV4CompressedComponent.commit_hicache_transfer()` 保持 no-op。

原因：

```text
FULL.value / FULL.host_value:
  由 FullComponent commit

SWA.value / SWA.host_value:
  由 SWAComponent commit

C4/C128/indexer compressed KV:
  可从 FULL indices 派生，不需要 node 级状态

compress_state:
  可从 SWA indices 派生，不需要 node 级状态
```

loadback 后，真正需要恢复的 device-side 可寻址状态由两个地方完成：

```text
FULL component:
  回填 node.component_data[FULL].value

SWA component + SWA allocator:
  回填 node.component_data[SWA].value
  重建 full_to_swa_index_mapping
```

`DeepSeekV4CompressedComponent` 不额外安装 fake SWA value，也不标记 state 为独立
device-ready。

### 5.6 不使用独立 DSV4_STATE component

当前方案不需要启用独立的 `DSV4_STATE` tree component。state 的 host/device indices
生命周期跟随 SWA component：

```text
SWA host/device indices exist
  -> corresponding state page rows can be located

SWA window absent
  -> state transfer should not run
```

这样 V4 的特殊逻辑集中在一个 component 中，UnifiedRadixTree 主流程不需要知道
C4/C128/indexer/state 的依赖关系。

## 待补充

- 单测设计。
