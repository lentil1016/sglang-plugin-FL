# Ascend NPU FlagGems 算子黑名单说明（sglang_fl）

## 1. 背景

sglang_fl 在 Ascend NPU（910C）上通过 FlagGems 做 **Layer-1 ATen 算子替换**：把 PyTorch ATen 算子替换为 FlagGems 的 Triton kernel。但 FlagGems 的部分算子在 Ascend 上存在缺陷，导致 sglang e2e（推理 TP=1 + serving TP=2，Qwen3）在初始化或推理阶段崩溃。

通过 `SGLANG_FL_FLAGOS_BLACKLIST` 环境变量把这些有缺陷的算子排除（回退到原生 torch_npu 实现），e2e 全部通过。本文档的黑名单已用 **e2e 移除法**逐个实测确认必要，精简到 6 个算子。

## 2. 崩溃现象

排查过程中依次遇到的崩溃：

| 阶段 | 崩溃点 | 错误 |
|---|---|---|
| KV 池初始化 | `allocator.py` / `zero_` / `zeros` | `coreDim 1082727 / 1171815 / 21654528 > 65535`（EE1003） |
| serving 初始化 | `index_put_` | 段错误（`PythonKernelHolder` segfault） |
| 采样 | `sampler.py:485 torch.multinomial` | Triton 编译失败（`exponential_` 的 `philox_seed` int32->int64） |
| overlap 调度 | `overlap_utils.py:24 future_token_ids_map[...]` | Triton 编译失败（`index` 的 bishengir-compile exit 1） |

## 3. 根因：两类缺陷

### A 类：coreDim 溢出

FlagGems 的 `zeros` / `zero_` kernel 在大 tensor 上启动时，Ascend Triton 后端把 `coreDim` 映射到元素数 N，而 Ascend 要求 `coreDim ≤ 65535`。sglang KV 池初始化对大 tensor 调 `torch.zeros` / `tensor.zero_()`，走 FlagGems 必崩。

**注意**：并非所有创建算子都崩。`arange` 虽在 `allocator.py:510 torch.arange(num_pages)` 对大池子调用，但 FlagGems arange kernel **不** coreDim 崩（直接测 21.6M 元素 + e2e 移除法均 OK）；`zeros_like` sglang 未在大 tensor 上调用。所以 A 类实际只命中 `zeros` / `zero_`。

### B 类：编译失败 / 段错误

部分 Ascend 专属 kernel 本身有缺陷，与 tensor 大小无关，靠运行暴露：

- `index`：`bishengir-compile` 退出码 1（Triton kernel 编不过）。
- `index_put_`：段错误（`PythonKernelHolder`）。
- `multinomial`：内部调用 `exponential_`，后者编译失败。
- `exponential_`：`fused_exponential_kernel` 循环携带变量 `philox_seed` 类型不一致（入口 int32，循环内 int64）。

## 4. 黑名单（6 算子，实测精简）

```
SGLANG_FL_FLAGOS_BLACKLIST=zeros,zero_,index_put_,multinomial,exponential_,index
```

| 算子 | 缺陷类 | 证据 |
|---|---|---|
| `zeros` | A coreDim | 移除后 e2e 崩 `zeros_kernel`（coreDim 1082727/1171815） |
| `zero_` | A coreDim | 移除后 e2e 崩 `zeros_kernel`（与 `zeros` 共用 kernel，`ops/zeros.py:48`） |
| `index_put_` | B 段错误 | 移除后 serving 初始化 segfault（`PythonKernelHolder`） |
| `multinomial` | B 编译失败 | 移除后 `sampler.py:485` 编译失败（经 `exponential_`） |
| `exponential_` | B 编译失败 | `philox_seed` int32->int64；`_ascend/ops/exponential_.py` |
| `index` | B 编译失败 | 移除后 `overlap_utils.py:24` 崩（`bishengir-compile exit 1`） |

**已验证不必要（移除后 e2e 仍 `Passed: 2, Failed: 0`）**：`zeros_like`、`arange`。这两个在早期清单里是预防性加入的，实测可去掉。演进路径：15 → 8 → 6。

## 5. 排除机制（关键）

黑名单排除是按**函数 `__name__` 逐个匹配**的（FlagGems `runtime/op_registrar.py:107`：`item[1].__name__ not in self.exclude_ops`）。

因此 `zeros`、`zero_`、`zeros_like` 是**三个独立函数**（都定义在 `ops/zeros.py`、共用同一个 `zeros_kernel`），但黑名单里写 `zeros` **不会**连带排除 `zero_` 和 `zeros_like` -- 必须各自单独列出。这是排查中踩过的坑：只黑 `zeros` 不够，`zero_` 仍会崩。

## 6. 移除法验证（reviewer 证据）

对每个算子单独从清单移除后跑完整 e2e（`tests/run.py --platform ascend --scope e2e`），判定必要性：

| 测试 | 黑名单（移除的算子未列入） | 结果 | 结论 |
|---|---|---|---|
| 移除 `zeros` | 其余 5 个 + zeros_like + arange | 崩 `zeros_kernel` | `zeros` 必要 |
| 移除 `zero_` | 其余 5 个 | 崩 `zeros_kernel` | `zero_` 必要 |
| 移除 `index_put_` | 其余 5 个 | 崩 segfault | `index_put_` 必要 |
| 移除 `multinomial` | 其余 5 个 | 崩编译失败 | `multinomial` 必要 |
| 移除 `exponential_` | 其余 5 个 | 崩编译失败 | `exponential_` 必要 |
| 移除 `index` | 其余 5 个 | 崩编译失败 | `index` 必要 |
| 移除 `zeros_like` | 6 算子 + arange | `Passed: 2, Failed: 0` | `zeros_like` 不必要 |
| 移除 `arange` | 6 算子 + zeros_like | `Passed: 2, Failed: 0` | `arange` 不必要 |
| **最小 6 算子** | `zeros,zero_,index_put_,multinomial,exponential_,index` | `Passed: 2, Failed: 0` | **最小集确认** |

## 7. 应用方式

### 临时（单次运行）
```bash
export SGLANG_FL_FLAGOS_BLACKLIST=zeros,zero_,index_put_,multinomial,exponential_,index
python tests/run.py --platform ascend --scope e2e
```
注意：必须 `export`，仅赋值不导出则子进程拿不到；换 shell 需重新 export。

### 持久化（建议）
写入 `sglang_fl/dispatch/config/ascend.yaml` 的 `flagos_blacklist` 字段，使所有 Ascend 场景默认生效（env 仍可覆盖）。

## 8. 排查方法

- 崩溃栈里 `current working operator name is X_kernel` 的 `X_kernel` 是 FlagGems Triton kernel 名（原生 torch_npu 不带 `_kernel` 后缀），据此定位是哪个 FlagGems 算子。
- Ascend 算子异步执行，栈可能不准；`ASCEND_LAUNCH_BLOCKING=1` 强制同步拿准确栈。
- 确认黑名单是否生效：看日志 `FlagGems enable (excluding: [...])`；若显示 `ALL ATen ops replaced` 则 env 未生效。
- **直接 Python 测单算子复现不了 e2e 崩溃**：`zero_` 在 e2e 里必崩，但 `flag_gems.enable()` 后直接 `t.zero_()` 在 21.6M 元素上 OK（`zeros`/`zeros_like`/`arange` 同样 OK）。coreDim 崩溃依赖 e2e 的 dtype/shape/上下文，判定算子必要性必须用真实 e2e 移除法，不能靠直接测。

## 9. 环境

- 硬件：Ascend 910C
- 软件：CANN 8.5.1、torch 2.11.0、torch_npu 2.11.0rc1、sglang 0.5.11、FlagGems
- 模型：Qwen3-0.6B（TP=1 推理）/ Qwen3-4B（TP=2 serving）
