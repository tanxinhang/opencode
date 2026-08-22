# P1 服务-时延理论桥：实现记录与深度审计

- 日期: 2026-08-22
- 依据: `advice/003.md`（P1 理论桥：R_T → reliable service → stopping-time delay）+ `advice/004.md`（P0.5 加固审计）
- 模块: `uav_otfs_isac/reliable_service_bridge.py`、`frids.simulate_frids_v2(bridge=True)`
- 门: `scripts/run_delay_bridge_gate.py` -> `results/delay_bridge_gate_p05.json`
- 测试: `tests/test_reliable_service_bridge.py`（11 项，全过）

## 更新: P0.5 加固记录(2026-08-22, advice/004)

`advice/004` 审计指出 P1 gate 的 7 处口径问题，已全部闭合（FRIDS-v2 全程未动）：

1. **stopped-process fill-forward**: `simulate_frids_v2(bridge=True)` 在目标停止后对
   `L/A/V/M/S/r/n_served` 保持终值成 `M_{t wedge T}`，不再回零（对应 optional
   stopping 下的停止鞅）。
2. **Freedman 统计口径**: 验证对象从 `P(M<=-eta)` 改为联合事件
   `P(M_t <= -eta, V_t <= v)`（确定性 (eta,v) grid，`eta` 用方差上界的分数、
   `v` 用 `V_up` 上界的分数），并新增 **time-uniform / line-crossing** 形式
   `P(存在 t: M_t<=-eta, V_t<=v)` ── 这正是 stopping time 的自然对象。全部
   scenario 点态+时一致 0 违例（max_ratio 0.25-0.41，界开始有意义而非恒 1）。
3. **量化 score drift 口径**: `L ̂=A ̃+M`，`A ̃` 用部署量化原子的 H1 条件漂移
   （非精确 KL），`delta_Q = |g - g ̃|` 单独记录（5-10%）；文档/命名已改。
4. **Pathwise stopping tail**: `stopping_tail_verify` 不再把 `mean(A), mean(V)`
   塞入非线性界，改为对每条路径算 `(A_q(t)-D_q)_+` 的指数项再取期望 ── 这才是
   Freedman 逐路径论证 + 期望的合法形式。实测所有案例 0 违例（cases 现在
   117-314，fill-forward 让前提真正激活）。
5. **eps_loc 重定义**: 原 eps_loc 的边界归一化爆炸是量纲伪影；新增
   `eps_loc_static_mean`（固定初始缺陷归一化，0.014-0.028）与
   `eps_loc_bottleneck`（local-vs-common CRN 瓶颈目标服务差，<=0.05，
   F0-G9A 的 ~1.8% 延迟对应的服务尺度）。
6. **Static-MD shadow gate**: 永不停止、固定 D/g、公共 shadow price +
   衰减步长 exp-gradient、T∈{20,40,80,160,320}。**关键发现**：per-UAV local
   dual 会 herd（全部 UAV 追同一目标，gap 不收敛 ~0.16）── 这本身就是
   eps_loc 的机制；理论对象是**单一公共价格**（定理 4.95/4.111），其 gap
   按 `O(sqrt(logQ/T))` 或更快衰减（实测 log-log 斜率 -0.63..-1.10，
   ≤ -0.5 即通过门）。
7. **cross-scenario verdict**: 现在遍历全部 8 场景（4 档 × 2 draw），
   不再只查 `16_8`。

结论维持: **FRIDS-v2 冻结**；定理 A（点态+时一致）与定理 B（8 场景）
数值成立；静态 MD 收敛率达理论速率或更快；瓶颈 eps_loc 小。

---

# 一、实现了什么

advice/003 建议的 P1 是：把 FRIDS-v2 从"实验上贴近边界"提升为"能够解释
为什么贴近边界，以及误差由哪些项决定"。本次落地为两个可证明定理的数值
验证链（严格尊重 advice/003 §一 的拆分要求，**不做一步式 R_T⇒ΔT**）：

- **桥-1（定理 4.110，服务→时延）**：部署规则的 owner LLR 增量在 H1 下的
  条件漂移恰为其携带的可靠信息，`E_1[Z|F_{t-1}] = Σ_i x_{iq,t} g_{iq,t}`
  （由定理 4.94 可靠信息恒等式保证），故

      L_q(t) = L_q(0) + A_q(t) + M_q(t),   M_q 鞅，L 精确分解。

  有限符号表（量化 + BSC + 擦除）⇒ 增量 a.s. 有界 `b_q`，**无需高斯
  shortcut**，直接用 Freedman 型鞅浓度不等式；进一步得到停时尾界

      P_1(T_q > t) ≤ β_q + exp[-(A_q(t)-D_q)² / (2(V_q(t)+b_q(A-D)/3))].

- **桥-2（定理 4.111，FRIDS→服务）**：需求归一化服务时间平均

      min_q (1/T) Σ_t r_q(t) ≥ z* − O(√(log Q / T)) − eps_loc,

  其中 `z*` = 定理 4.95 的静态需求归一化 LP（多项式，无枚举），`eps_loc`
  为分布式信息损失（投递擦除 + 局部对偶分歧）。

# 二、记录内容（`simulate_frids_v2(bridge=True)`）

按 (run, cycle, target) 记录：owner LLR `L`、累计可预测可靠服务 `A`
（部署量化符原子在 H1 下的精确条件均值 × 真实投递概率）、`g`-记账的
未量化 `A_raw`、可预测条件方差 `V`、鞅残差 `M = L - A`、实际送达的
`i_plus` 服务 `S`、需求归一化服务 `r_pred/r_real`、并发服务数
`n_served`、停时 `T`、假设 `H`。`bridge=False` 默认路径逐字节不变
（有测试锁定）。

# 三、数值结果（`results/delay_bridge_gate.json`，每档 2 场景，500 runs）

| 尺度 | 分解误差 | Freedman 违例 | 停时尾界违例 | quant.-gap | z*−min_r^st |
| --- | --- | --- | --- | --- | --- |
| (6,3) | 1e-15 | 0/480 | 0 | 5–10% | 0.026–0.067 |
| (8,4) | 1e-15 | 0/640 | 0 | 6% | −0.03–0.13 |
| (12,6) | 1e-15 | 0/960 | 0 | 4–5% | −0.05–0.07 |
| (16,8) | 1e-14 | 0/1280 | 0 | 5% | −0.10–0.26 |

## 门判定（cross-scenario, 16_8）

`decomposition_ok = true, stopping_tail_ok = true, service_gap_ok = true` →
**FRIDS-v2 冻结**；定理 A 数值成立，定理 B 的 `z*−min r^st` 落在
`√(logQ/T) ≈ 0.166–0.23` 加允许量内（唯一超出的 16,8-s1 为 0.261，仍
在 `√ + 0.05` 内）。

# 四、深度审计（问题与边界）

1. **停时尾界很少激活（最重要的诚实发现）**。前提 `A_q(t) ≥ D_q` 在每个
   场景只激活 1–13 例（目标在 ~2.6–4 周期就被实现漂移停住，可预测服务
   通常还没越过需求缺陷）。因此定理 4.110 的尾界是**结构性充分界**：
   它证明"服务不足 + 波动 + 漏检预算"是仅有的失败通道，而不是当前工作
   点的紧预测器。这正是 advice/003 §九 Case B 的诚实处理方式：论文不得
   声称它预测中可行区的实测时延；它只在可行性边界（实验 Γ<1，F0-G6）
   才成为绑定解释。

2. **分解精确到浮点精度（1e-15）是构造性的**。`M = L - A` 按定义使
   `L = A + M` 恒成立，因此分解误差验证的是"记账正确 + 漂移定义正确"
   （E_1[Z|F] = 部署原子的条件均值），而不是一个先验的独立定理。真正
   的内容是：鞅残差零均值（|mean| ≤ 1 nats）、Freedman 尾界 0 违例。

3. **量化修正审计（advice/003 §四 的判据）**。`A_raw`（用未量化 g 记账）
   与 `A`（部署 5-bit 原子漂移）的相对差仅 4–10% → **有限阈值/量化
   修正太小，不要做 finite-threshold FRIDS，FRIDS-v2 保持冻结**。这是
   advice/003 问的"要不要上线 finite-threshold 版本"的量化答案：不要。

4. **eps_loc 的量纲陷阱（审计发现的自身缺陷并修复）**。最初用当前缺陷
   `D_q(t)+eps` 归一化服务比，在决策边界 D→0 处数值爆炸（5–14），这不
   是真实的分布式损失而是归一化伪影。已修复：`eps_loc_static_mean` 用
   固定初始缺陷归一化，实测仅 0.014–0.028（投递擦除 + 量化），边界归一
   化值保留为诊断并标注伪影性质。

5. **非主张（写进 FORMAL_PROOFS 定理 4.110/4.111）**：
   - 尾界是充分非紧条件（Case B）——不声称预测中可行区时延；
   - 定理 4.111 在静态归一化下验证，owner 记账的 r_real 可超调度服务
     （边界附近短缺陷效应），不是完整序贯 MDP 最优性声明；
   - **不做一步式 R_T⇒ΔT**，两座桥分开陈述，论文组合时只用
     "两桥 + 实验 Γ∈[0.75,1]" 这一保守链。

# 五、与 advice/003 判定规则的对齐

- §九 Case A/B/C：本结果属 **Case B/C 组合**——服务桥（定理 4.111）闭合
  （Case A 性质）、停时尾界只作结构界（Case B 边界）、不强行写一步式
  桥（Case C 规则）。这是 advice/003 预期的最稳健出口。
- §四 问号"finite-threshold FRIDS 要不要上线"：量化修正 4–10% ⇒ 不上线，
  v2 冻结（数值结论）。
- §八（时变信道用 dynamic regret，不发明 mobility-aware FRIDS）：本次
  未展开，`bridge` 记录已内建 mobility 缩放（`mfac` 进入漂移/方差），
  为后续 P3 外推留好了接口。

# 六、文件清单

- `uav_otfs_isac/reliable_service_bridge.py`（新）：`freedman_tail`,
  `stopping_tail_bound`, `static_relaxation_optimum`,
  `martingale_decomposition`, `stopping_tail_verify`,
  `normalized_service_time_average`, `relative_error_bound`。
- `uav_otfs_isac/frids.py`：`simulate_frids_v2` 增加 `bridge=True` 记录。
- `scripts/run_delay_bridge_gate.py`（新）：P1 门。
- `tests/test_reliable_service_bridge.py`（新）：8 项测试。
- `docs/FORMAL_PROOFS.md`：定理 4.110/4.111 及实现-测试追踪。
- `docs/THEORY_DEVELOPMENT.md`：§1.27 + 索引行。
- `results/delay_bridge_gate.json` / `results/delay_bridge_gate_p05.json`：门输出。

# 七、FORMAL_PROOFS.md 丢失与重建记录（诚实披露）

2026-08-22 审计发现：上一轮提交前执行的 `git checkout .`（在 `git add -A`
之前）把工作区未提交的 `docs/FORMAL_PROOFS.md`（146,913 字节 / ~2925 行，
含 FRIDS 理论章节 5B–5D 定理 4.94–4.109）回退到暂存区旧版（1668 行），
并丢失本桥新增的定理 4.110/4.111。git 对象库与磁盘均无备份。

**这是执行失误，与理论/门判定无关；P0.5 结论不变。** 已按以下信源重建：

- 4.91–4.98、4.109：本会话上一轮对工作区的**逐字读取**；
- 4.99–4.108：按 `THEORY_DEVELOPMENT_HISTORY.md` 阶段 34/35 的描述**忠实
  重建**，并在文中显式标注 `(Reconstructed from ...)`；
- 4.110/4.111：本轮加固后的全文（5E 节）。

重建后的 `docs/FORMAL_PROOFS.md`（2721 行，139,152 字节，无遗留字符错误）
已含 5A–5E 全部章节与实现-测试追踪行，并提交。