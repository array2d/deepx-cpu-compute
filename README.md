# deepx-cpu-compute

kvlang 的 **CPU tensor 计算 main-runtime**。

## 定位

runtime 核心只有一套——`kvlang/runtime`（C，唯一标准实现）。其余 runtime 一律是**扩展运行时**：嵌入 C 核心，套用其取指-解码-执行、native 派发与正典 codec，只在自己的 **myrwircaps** 里追加己方 opcode 的解释（见 kvlang `stdlib/kvlang/spec/05-runtime语义/`）。

deepx-cpu-compute 是 CPU 侧的这样一个 main-runtime：

- **自身只完成 tensor 运算**——myrwircaps = `tensor.*` 算子表（`tensor.add` / `tensor.matmul` / `tensor.sum` …），用 deepx-core kernel（OpenBLAS + Highway SIMD + OpenMP）就地兑现。
- **其余 rwir 交给它调用的 runtime-c**——控制流（call/return/br/goto）、native builtin、用户函数由嵌入的 C 核心解释；非 tensor 的扩展 rwir（print/json/…）经 handoff 交对应扩展进程。

opcode 契约与 numpy / op-gpu 一致（`tensor.<op>`），同一 rwir 无论落哪个 runtime 行为一致。

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
register()                               # 每个 tensor.* opcode 写 /lib/<op> 签名
vid = Bootstrap(entry)
loop:
  rc = ExecuteVthread(rt, vid, &pc)      # C 核心跑到遇 ext rwir 冒泡
  rc==0 → done; rc<0 → error
  op = Params(pc)
  op ∈ myrwircaps → 就地跑 CPU kernel（零拷贝）+ NextPc
  else            → Handoff 交外部扩展
```

**tensor 零拷贝**：`ResolveReadPath` 拿容器路径 → `kvspaceGet` 取 TLV → `DecodeHead` 定位 body → 在该地址原位建 `Tensor<T>`（借用 data、`deleter=nullptr`）跑 kernel；写经 `TlvEncode` + `WriteNewPlace`/`WriteInPlace`。tensor data 直接落 kvspace-c 的 shm。

## myrwircaps（CPU tensor 运算表）

已实现（deepx-core kernel，零拷贝）：

| 类 | opcode | dtype |
|----|--------|-------|
| elementwise | `tensor.add` `tensor.sub` `tensor.mul` | float32/64、int32/64 |
| elementwise | `tensor.div` | float32/64（整型 SIMD 无除法） |

规划（按 issue #1 落地顺序补齐）：

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
及其 `/usr/include` 头；`libopenblas-dev`（openblas-pthread）、`libhwy-dev`、`libjemalloc-dev`、
`libyaml-cpp-dev`、`nlohmann-json3-dev`、`pkg-config`、OpenMP。
`-DDEEPX_CPU_BUILD_TESTS=ON` 另建 deepx-core kernel 单元测试（默认关）。

## 运行

```bash
export KVSPACE=redis://127.0.0.1:6379          # 或 shm:///tmp/xxx
deepx-cpu-compute lib/tutorial/01-elementwise.kv   # layout+运行 .kv
deepx-cpu-compute <entry>                       # 运行已入库入口
DEEPX_DUMP=1 deepx-cpu-compute file.kv          # 额外把每次 tensor.* 结果打到 stderr（结果为帧内局部，return 后清）
```

## 目录

- `deepx-core/` — 共享核心：shape / dtype / tensor / tf 工厂。
- `src/deepx/tensorfunc/` — CPU 计算 kernel（elementwise / matmul / reduce / changeshape / init / io）。
- `src/deepx/tf/` — 算子定义。
- `src/main-runtime/` — main-runtime：`main.cpp` 驱动循环 + tensor myrwircaps 派发；
  `tensor_bridge.hpp` kvspace body ↔ `Tensor<T>` 零拷贝桥；`hwy_compat.hpp` 补 highway 1.0.7 缺失的 `IsAligned`。
- `lib/tutorial/` — kvlang tensor 示例（一致性测试）。
