# deepx-cpu-compute

kvlang 的 **CPU tensor 计算 main-runtime**。

## 定位

runtime 核心只有一套——`kvlang/runtime`（C，唯一标准实现）。其余 runtime 一律是**扩展运行时**：嵌入 C 核心，套用其取指-解码-执行、native 派发与正典 codec，只在自己进程内的 **myrwircapstable**（一张「key: opcode 串, id: handler 编号」表）里追加己方 opcode 的解释（见 kvlang `stdlib/kvlang/spec/05-runtime语义/`）。

deepx-cpu-compute 是 CPU 侧的这样一个 main-runtime：

- **自身只完成 tensor 运算**——myrwircapstable = `deepx/miaobyte·*` 算子表（`deepx/miaobyte·add` / `deepx/miaobyte·matmul` / `deepx/miaobyte·sum` …），用 deepx-core kernel（OpenBLAS + Highway SIMD + OpenMP）就地兑现。
- **其余 rwir 交给它调用的 runtime-c**——控制流（call/return/br/goto）、native builtin、用户函数由嵌入的 C 核心解释；别的 runtime 的 rwir（print/json/…）经 handoff 交对应进程。

opcode 按 `/lib/deepx/<作者>·<op>` 命名空间组织：同一 tensor 运算可有多套实现，各占一支（`deepx/miaobyte·add`，未来 `deepx/cblas·add`…），由调用点选定。执行只查本进程 myrwircapstable；`/lib/<opcode>` 注册仅供分布式调度，本进程派发不依赖它。

## 架构

嵌入三套已安装 C ABI：

| 头 | 库 | 用途 |
|----|-----|------|
| `kvlang_runtime.h` | `libkvlang_runtime.so` | 核心执行：`Connect` / `KvspaceHandle` / `Bootstrap` / `ExecuteVthread` |
| `kvlang_rwirext.h` | （同上） | rwirext 宿主：`Register` / `Params` / `ResolveRead(Path)` / `ResolveWrite` / `NextPc` / `Handoff` |
| `kvspace/kvspace.h` | `libkvspace.so` | KV 存取：`Get`（借用零拷贝读）/ `WriteNewPlace` / `TlvEncode` / `DecodeHead` |

**驱动循环**（对齐 `kvlang/runtime-rs`）：

```
rt = kvlangRuntimeConnect(dsn)
kv = kvlangRuntimeKvspaceHandle(rt)      # 复用同一句柄（durable 惰性 flush 只在同句柄内相干）
register_myrwircaps()                    # 每个 opcode 写 /lib/<op> 签名（供分布式调度）
vid = Bootstrap(entry)
loop:
  rc = ExecuteVthread(rt, vid, &pc)      # C 核心跑 native/控制/用户 rwfunc，遇 ext rwir 冒泡
  rc==0 → done; rc<0 → error
  连续批处理 pc 起、命中 myrwircapstable 的 opcode：就地跑 CPU kernel（零拷贝）+ NextPc
  停在非本表的指令 c：
    c ∈ 别的 runtime 的 ext rwir → Handoff（跨 runtime 协作，暂无）
    否则（native/控制/帧结束）    → 写回 pc，ExecuteVthread 续跑
```

**tensor 零拷贝**：`ResolveReadPath` 拿容器路径 → `kvspaceGet` 取 TLV → `DecodeHead` 定位 body → 在该地址原位建 `Tensor<T>`（借用 data、`deleter=nullptr`）跑 kernel；写经 `WriteNewPlace`/`WriteInPlace` 拿可写 body 指针。

**内存模型**：tensordata 一律活在 kvspace（shm 后端）分配的内存里，`Tensor<T>` 只借用该地址（`newer/deleter/copyer` 全为 `nullptr`），不经 deepx-core 的内存池。故本 runtime **不依赖 jemalloc**，deepx-core 的 `mempool`/`tensorlife`/`io` 均不参与编译。tensor 的类型即 kvspace langtype `[dims]dtype`（如 `[256,256]float32`），不引入额外的 “tensor” 类型名。

## myrwircapstable（CPU tensor 运算表）

已实现（deepx-core kernel，零拷贝）：

| 类 | opcode | dtype |
|----|--------|-------|
| elementwise | `deepx/miaobyte·add` `deepx/miaobyte·sub` `deepx/miaobyte·mul` | float32/64、int32/64 |
| elementwise | `deepx/miaobyte·div` | float32/64（整型 SIMD 无除法） |

规划（按 issue #1 落地顺序补齐；下列 opcode 均属 `deepx/miaobyte·` 命名空间，未来 cblas 后端同名挂 `deepx/cblas·`）：

| 类 | opcode |
|----|--------|
| elementwise | scalar 变体、`sqrt` `pow` `log` `exp` `sin` `cos`、`max` `min`、`neg` `abs`、比较 `eq` `lt` `gt` |
| matmul | `matmul` |
| reduce | `sum` `prod` `mean` `max` `min` |
| changeshape | `reshape` `transpose` `concat` `broadcastTo` |
| init | `zeros` `ones` `full` `arange` `uniform` `normal` |

原则：executor 只实现基础 op，不实现可由基础 op 组合的 op（如 relu 由 max 组合）。

## 构建

```bash
bash build.sh    # CMake，产物 /tmp/deepx/cpu-compute/<os>-<arch>/deepx-cpu-compute-<os>-<arch>
```

依赖（Linux apt 包名）：已装 `/usr/lib` 的 `libkvlang_runtime.so` / `libkvspace.so` / `libkvlanglayout.so`
及其 `/usr/include` 头；`libopenblas-dev`（openblas-pthread）、`libhwy-dev`、OpenMP。
（第三方依赖仅此三项——序列化与内存都走 kvspace，不依赖 jemalloc / yaml-cpp / nlohmann-json。）

## 运行

```bash
export KVSPACE=redis://127.0.0.1:6379          # 或 shm:///tmp/xxx
deepx-cpu-compute lib/tutorial/01-elementwise.kv   # layout+运行 .kv
deepx-cpu-compute <entry>                       # 运行已入库入口
DEEPX_DUMP=1 deepx-cpu-compute file.kv          # 额外把每次算子结果打到 stderr（结果为帧内局部，return 后清）
```

## 目录

- `deepx-core/` — 共享核心：shape / dtype / tensor 类型系统与 shape 推导（编入 STATIC 库），tensorfunc 泛型 dispatcher 接口。
- `src/deepx/tensorfunc/` — CPU 计算 kernel 实现（`*_miaobyte` / `*_cblas`：elementwise / matmul / reduce / changeshape / init）。
- `src/main-runtime/` — main-runtime：`main.cpp` 驱动循环 + myrwircapstable 派发；
  `tensor_bridge.hpp` kvspace body ↔ `Tensor<T>` 零拷贝桥；`hwy_compat.hpp` 补 highway 1.0.7 缺失的 `IsAligned`。
- `lib/tutorial/` — kvlang tensor 示例（一致性测试）。
