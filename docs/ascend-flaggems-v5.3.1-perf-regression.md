# FlagGems v5.3.1 Ascend serving 解码性能劣化排查报告

> 面向：FlagGems / FlagOS 项目 owner。

## 结论

**根因**：FlagGems v5.3.1 把 `F.linear` 路由到通用 `flag_gems.ops.linear`，该实现**每次 launch 有 ~15ms 的 per-launch 同步开销**（kernel 已缓存、非重编译；py-spy 所见 `OpCommand::Run -> aclrtPointerGetAttributesImpl -> GetRtChipType` 同步点）。v5.0.0 走 ascend `addmm`（0.16ms/call），v5.3.1 的 `mm`/`bmm` 走原生（0.08ms/call），均无此开销。decode 每 token ~250 次 linear × 15ms ≈ 3.85s/token ≈ 0.26 tok/s（与实测 0.25 tok/s 吻合）。

**Fix（已端到端验证）**：`SGLANG_FL_FLAGOS_BLACKLIST` 加 `linear`（5-op：`mul,index_put_,_index_put_impl_,index,linear`），把 `F.linear` 路由到原生（0.27ms/call，不崩）。e2e 实测 `[run] Passed: 2, Failed: 0`、serving smoke `5 passed in 279s`（未修 778s+）。**v5.3.1 现可在 Ascend 上使用，不必回退 v5.0.0。**

**需 owner 修复**：通用 `ops.linear` 在 NPU 上的 per-launch ~15ms 同步开销（v5.0.0 的 ascend `addmm` 与原生 `mm` 均无）。要么消除该 per-launch 同步，要么 ascend 后端恢复 `linear` 的专用实现（像 v5.0.0 的 addmm）。

---

## 1. 现象

| 项 | 版本 |
|---|---|
| 硬件 / CANN | Ascend 910C 64GB / 8.5.1 |
| torch / torch_npu / triton-ascend | 2.11.0 / 2.11.0rc1 / 3.2.1 |
| sglang / FlagGems | 0.5.11 / **v5.3.1（劣化）vs v5.0.0（正常）** |
| 模型 | Qwen3-4B（serving，TP=2） |

- v5.0.0：serving 解码 ~20 tok/s，e2e ~87 s。
- v5.3.1：serving 解码 **0.25–0.33 tok/s**，e2e **~778 s**（慢 ~9× 端到端 / ~60–80× 吞吐）。
- 劣化主要在 **serving 解码**（每 token 多次小算子调用）；inference（TP=1，大 batch matmul）受影响小，per-launch 开销被摊薄。

---

## 2. 根因：v5.3.1 通用 `ops.linear` per-launch ~15ms

微基准（bf16, NPU, decode 量级 shape `H=2560`，同 shape 重复 100 次取均，已 warmup 编译）：

| op | v5.0.0 (ms/call) | v5.3.1 (ms/call) | v5.3.1 路由 |
|---|---|---|---|
| `mm` | 0.083 | 0.081 | 原生 |
| `bmm` | 0.076 | 0.072 | 原生 |
| `scatter_` | 0.086 | 0.088 | flag_gems |
| **`linear`（F.linear）** | **0.162** | **15.429** | flag_gems（通用） |

- v5.3.1 的 `linear` **15.4 ms/call**，是 v5.0.0（0.162）的 ~95×、同版 `mm`（0.081）的 ~190×。其余算子两版基本一致。
- **非 per-call 重编译**：10 次同 shape `linear` 调用后 triton cache 条目数不变（16 -> 16），kernel 已缓存复用。
- **是 per-launch 同步开销**：py-spy 显示 v5.3.1 `linear` 每次 launch 过 `at_npu::native::OpCommand::Run -> aclrtPointerGetAttributesImpl -> GetRtChipType`。15ms/call 与该 per-launch 同步点吻合。v5.0.0 的 ascend `addmm` 与 v5.3.1 原生 `mm` 均无此开销。
- **decode 每 token linear 调用数**：Qwen3-4B ~36 层 × ~7 次 linear/层 ≈ 250 次/token。
  - v5.3.1：250 × 15.4ms ≈ **3.85 s/token ≈ 0.26 tok/s**（吻合实测 0.25）
  - v5.0.0：250 × 0.162ms ≈ 40ms/token ≈ 25 tok/s（吻合实测 ~20）

**为什么只有 `linear` 中招**：`mm`/`bmm`/`matmul` 在 v5.3.1 走原生（无 per-launch 开销）；`F.linear` 在 v5.3.1 被通用 `flag_gems.ops.linear` 接管（v5.0.0 是 ascend `addmm`）。即 v5.3.1 给 ascend 后端改用了通用 linear 实现，而该实现每次 launch 触发 `GetRtChipType` 同步。

---

## 3. Fix + 验证

**Fix**：`SGLANG_FL_FLAGOS_BLACKLIST` 加 `linear` -> `F.linear` 路由到原生 torch_npu。

- 微基准（v5.3.1，`unused=['linear']`）：`linear` 从 15.4 降到 **0.266 ms/call**（~58× 提速，与 v5.0.0 的 0.162 同量级）；record log 中 `linear` 不再出现（确认路由原生）；**不崩**。
- 端到端（v5.3.1，5-op 黑名单 `mul,index_put_,_index_put_impl_,index,linear`，`env -u FLAGCX_PATH`）：
  ```
  [run] Passed: 2, Failed: 0
  serving smoke: 5 passed in 279.68s   # 未修 778s+，test_endpoint[chat] PASSED
  ```

---

## 4. 已排除的路径（排查中走过、经验证非根因）

为避免 owner 重复走弯路，简列已排除项（详细证据见 git 历史版本的本文档 / `docs/ascend-flaggems-v5.3.1-issues.md`）：

- **FlagCX 通信器**：劣化在单 server 解码，不涉及通信；两版均走 HCCL。
- **`floor_divide` 每 token 重编译**（旧 `issues.md` P4 的归因）：A/B 加/不加 `floor_divide` 黑名单 e2e 无差别（778 vs 793s）。旧 P4 称"黑名单对 floor_divide 不生效"亦错误--`aten_patch_list` 为空，黑名单生效。floor_divide 确实每 token 重编译，但只占劣化很小部分。
- **`pointwise_dynamic` 的 stride `int`->`tl.constexpr` 改动**：微基准两版都按 shape 重编译，v5.3.1 甚至略快，非主因。
- **attention/matmul routing**：record log 实测 v5.3.1 反而把 `mm`/`bmm`/`matmul` 走原生（比 v5.0.0 更少 flag_gems），`sdpa` 也走原生；"v5.3.1 新把 attention/matmul 塞进 flag_gems"证伪。`linear` 是唯一两版都走 flag_gems 但实现不同的算子 -> 真凶。
- **`scatter_` 路由/CPU 回退**：两版都走 flag_gems 通用 `_scatter_`；ascend `scatter.py` 的 `scatter_add_no_atomic` CPU 回退只在 `scatter_(reduce="add"/"multiply")` 触发，penalty 路径不踩。
- **黑名单 `scatter_`/`gather` 走原生作 workaround**：会让 server 崩（`peer closed connection` + `Connection refused`），不可用（但 `linear` 黑名单不崩，已验证）。

---

## 5. 复现

- **宿主机**：`bm-jn-zs-bj2-910C-64G-10-115`（910C 64G），`tmux attach -t ascend`。
- **容器**：`fl500`（v5.0.0）、`fl531`（v5.3.1），均挂 `/root/sglang-plugin-FL` -> `/workspace`。**该机同一时刻只有一个容器能持 NPU 驱动**，切容器须 `docker stop` 一个再 `docker start` 另一个。
- **复现劣化**（v5.3.1，慢）：
  ```
  SGLANG_FL_FLAGOS_BLACKLIST=mul,index_put_,_index_put_impl_,index \
    env -u FLAGCX_PATH python tests/run.py --platform ascend --scope e2e
  # -> Passed: 2，但 ~778s，serving 解码 0.25 tok/s
  ```
- **Fix 验证**（v5.3.1，快）：黑名单改成 `mul,index_put_,_index_put_impl_,index,linear`，同命令 -> `Passed: 2`，serving 279s。
- **linear 微基准**（两版各跑对比 ms/call）：`docker exec fl531 python3 /workspace/_bench3.py`。
- **对照**（v5.0.0，正常）：`docker stop fl531 && docker start fl500`，v5.0.0 用 6 算子黑名单 `zeros,zero_,index_put_,multinomial,exponential_,index`，~87s。

---

## 6. 给 owner 的建议

1. **【立即可用 workaround】Ascend 镜像 v5.3.1 黑名单加 `linear`**（5-op：`mul,index_put_,_index_put_impl_,index,linear`）。已端到端验证 `Passed: 2`、serving 279s。可不回退 v5.0.0、继续用 v5.3.1（CUDA 不受影响）。
2. **【owner 修 FlagGems】通用 `flag_gems.ops.linear` 在 NPU 上的 per-launch ~15ms 同步开销**。证据：同 shape 100 次 15.4ms/call、cache 不增（非重编译）、py-spy 见 `OpCommand::Run -> aclrtPointerGetAttributesImpl -> GetRtChipType`。需查清为何通用 linear 每次 launch 触发该同步（v5.0.0 的 ascend `addmm` 与原生 `mm` 都没有），要么消除该同步、要么 ascend 后端恢复 `linear` 的专用实现。
3. **【附带回归】黑名单 `scatter_`/`gather` 走原生会崩**：v5.3.1 `flag_gems.enable()` 后原生 `scatter_`/`gather` 崩（`linear` 黑名单不崩，可对照）。本身是 v5.3.1 回归，建议一并查。
