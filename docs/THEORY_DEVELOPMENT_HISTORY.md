# 理论循序发展记录

- 日期: 2026-08-17(记录截至当日;gate 日期以结果 JSON/提交历史为准,本文只给顺序)
- 定位: 按**发展阶段**(gate 顺序)记录理论如何一步一步建立(动态视图);当前完整框架见 `docs/THEORY_DEVELOPMENT.md`
- 定理编号锚定: `docs/FORMAL_PROOFS.md`;实验数字锚定: `docs/SYSTEM_RESEARCH_REPORT.md`

---

## 0. 如何阅读

每条记录含五个要素: **动机**(该阶段要解决什么问题)、**理论结果**(定理/引理号)、**实现**(模块/脚本)、**验证**(gate/测试,通过或负结果)、**对后续的影响**。三条发展主线贯穿全程:

1. **精确性 → 可扩展性**: 从精确枚举(2^R)到 Pareto 剪枝、B&B、注水、NOMP refine;
2. **集中 → 分布式**: 从单一融合中心到逐级去中心化共识、对等多数、分布式松弛;
3. **名义 → 鲁棒**: 从干净信道到 BSC/擦除/通信模糊/系数误差的最坏情形保证。

---

## 1. 阶段 0: OTFS 物理层与前端(Gate 0 / G0-C)

- 动机: 建立 DD 域物理基础与接收前端,避免"OTFS 只是标签"。
- 结果: 分数多普勒 sinc^2 泄漏、DD 网格矩匹配证据、MF/CFAR 前端、逐路径 Fisher 型协方差;`waveform`/`otfs_physical`/`front_end`。
- 验证: G0-C 前端门控(集成帧数 4 仅正式运行);结论: OTFS 保留为场景背景,理论不依赖波形级细节。

## 2. 阶段 1: 证据校准与报告链路闭合(G1-A / G1-B)

- 动机: 证据矩估计是否可信;量化→BSC→擦除→重建矩链路是否闭合。
- 结果: G1-A 形式校准(10k 试次,训练/测试几何分离);G1-B 报告信道闭合(50k MC)。
- 影响: 确立"通信损失进入 H0/H1 矩"的价值模型,成为全论文的理论地基。

## 3. 阶段 2: 条件排序与贪心验证(G1-C / G1-D)

- 动机: 条件相关排序是否真有价值;贪心相对 Oracle 的行为。
- 结果: Conditional Greedy 对 Static ID Top-K 胜率 77.5%–83.1%;G1-A 中 Exact-`P_D`-增益贪心 Spearman 0.996(偏转代理仅 0.588)。
- 影响: 确立"偏转代理不足、精确 P_D 增益排序必要",引出 G4。

## 4. 阶段 3: 系统级扫描与负结果(G2)

- 动机: 全系统级(非单 gate)对比与稳健性;防止过拟合特定配置。
- 结果: 公平系统扫描、算法负门控、相关扫描、非饱和压力、缩放、资源公平;老几何-Doppler 贪心被证伪(DBSCAN 系 100% vs 83% @ 1.2-7.4ms),降级为消融基线。
- 影响: 主算法的候选集收敛;统计证据升级为 500 种子双侧 t/Wilcoxon + Holm 校正。

## 5. 阶段 4: 融合理论(Gate G3)

- 动机: 收到集合确定后,线性融合权重如何取;集合单调性从何而来。
- 结果: KKT 单参数族包含全局线性最优(`P_D > 0.5`,定理 2.3)、集合单调(定理 2.4)、比例协方差闭式(定理 2.5)。
- 影响: 融合从"启发式权重"变为"可证明最优族";`gaussian_pd_closed_form` 成为全系统数值内核。

## 6. 阶段 5: 期望-P_D 理论(Gate G4)

- 动机: 擦除随机性下选择目标如何定义;贪心是否有界。
- 结果: 期望保单调(3.1)、有界区域次模(3.4)→ 基数贪心 `1 - 1/e`;期望 `P_D` 增益贪心 B=20 最差目标 +7.56pp。
- 影响: 主选择目标正式定为 `E_PD(q, S_q)`;为 G8 精确选择铺路。

## 7. 阶段 6: RIS 信道、量化与部署(Gate G5)

- 动机: RIS 增益/量化/布放搜索的边界。
- 结果: 加性功率增益无害(4.1)、量化损失界(4.2)、网格界(4.3)、Lipschitz B&B epsilon 证书(4.4-4.5)。
- 影响: 部署搜索获得可验证证书;RIS 降格为"几何感知归一化功率增益模型的应用实例"。

## 8. 阶段 7: 精确选择证书(Gate G8, 4.7-4.7G)

- 动机: 贪心无界时,预算/配额选择能否精确。
- 结果: 等成本配额(4.7)、异构成本预算(4.7A)、max-min(4.7B)、Pareto 剪枝(4.7C)、最小成本 B&B(4.7D)、缩放证书(4.7E)、Cauchy 上界(4.7G)。
- 影响: **精确性成为系统的核心卖点之一**;审稿 P0-4/P0-7 后补"目标可分+加性成本"假设与组合/离散化边界。

## 9. 阶段 8: 孔径、注水与证书(G9-G18)

- 动机: 把资源从"报告选择"扩展到 RIS 孔径分配与布放。
- 结果: 孔径守恒/上升(4.8-4.9)、孔径-开销权衡(4.10)、闭式最优(4.11)、max-min 注水(4.12)、系统级局部最优(4.14)、单/多块证书(4.15-4.16)、联合布放-分配(4.17)、有限终止与复杂度(4.24)、Q>3 缩放(4.25)。
- 影响: 明确"白盒、精确信息、局部证书"方法论;G18 成为"非神经网络"主张的代表。

## 10. 阶段 9: 去中心化阶梯(G19-G25)

- 动机: 融合中心是单点;逐步去掉中心后性能如何降级。
- 结果: 比特粒度(4.18)、分布式门限(4.19)、对等多数(4.20)、多跳(4.21)、公共故障(4.22)、扩展(4.23)。
- 影响: 引出计数理论(Poisson-binomial)在阶段 13 的精确化。

## 11. 阶段 10: 复杂度升级阶梯(G26-G31)

- 动机: 静态几何/单 RIS/固定速率太"玩具";按 6G 场景逐项升级。
- 结果: 移动+时变阻塞(4.26)、多 RIS(4.27-4.28)、可变速率报告(4.29-4.30)、软硬混合融合(4.31)、干扰注入(4.32)。
- 影响: 场景从静态升级到移动/多 RIS/变速率,每步保留正式声明。

## 12. 阶段 11: 干扰、孔径构型与零陷(G32-G39)

- 动机: 无干扰/无杂波假设过强;阵列构型与干扰抑制。
- 结果: 空间 INR(4.33)、多源叠加(4.34)、UPA vs ULA(4.35)、零陷标量化(4.36)、量化零陷上升(4.37)、联合零陷-布放(4.38)、QoS 松弛(4.39)、分布式松弛(4.39 门控)。
- 影响: 干扰敏感性成为受控实验而非缺失维度。

## 13. 阶段 12: 共识可行性与奇偶边界(G40-G43)

- 动机: 共识何时真的可行;多数投票的可行性判定。
- 结果: 稀缺比特优势(4.40)、奇偶边界(4.41)、优化门限(4.42)、**精确泊松-二项可行性**(4.43)、精确最小多数(4.43A,投票计数非单调)。
- 影响: 共识可行性从"估计"变"精确判定"(M=6 起);成为阶段 14 架构切换的依据之一。

## 14. 阶段 13: 信息预算与架构切换(G44-G50)

- 动机: 集中 vs 分布式到底由什么决定;能否定量。
- 结果: 预算单调(4.44)、朴素律失效(4.45)、`rho_exact` 坐标(4.46)、双分支切换(4.47)、逐目标支配(4.48)、软比特再分配(4.49)、模式上升(4.50-4.51)、迟滞界(4.56-4.57)。
- 影响: "信息预算"成为统一解释框架;B=8/12 切换共识、B>=16 回归集中,均可精确计算。

## 15. 阶段 14: 移动与预测(G51-G54)

- 动机: 随机移动 + RIS 重配延迟下的最差时序性能。
- 结果: 模式上升推向最差时序 QoS(4.51)、AR(1) 条件均值预测(4.52)、h 步 MMSE(4.53)、**负结果**: 协方差感知期望增益代理在量化下非最优(4.54-4.55)。
- 影响: 确立 MMSE 预测为主、期望增益代理被否定;负结果本身成为论文的诚实边界。

## 16. 阶段 15: 鲁棒机会约束与信道单调(G30-E 与 4.58-4.64)

- 动机: 信道参数不确定时如何给出最坏情形保证。
- 结果: 最坏情形机会约束分配(4.58)、BSC 退化 ROC 支配(4.59)、擦除单调(4.60)、移动包络(4.61/4.61A)、独立模糊标量 DP(4.62)、复杂度(4.63)、物理链路模型(4.64);G30-E 精确率档案 B28/B40 修正。
- 影响: 系统从"名义最优"升级为"最坏情形精确";鲁棒端点方法成型。

## 17. 阶段 16: 联合功率-比特与通信模糊(4.65-4.73)

- 动机: 感知功率与通信比特联合分配;通信参数区间不确定。
- 结果: 阈值可行性复杂度(4.65)、选项集含基线(4.66)、向量化枚举(4.67)、信道解耦(4.68)、通信感知得分(4.69)、**端点归约**(4.70/4.70A)、鲁棒联合 DP 精确(4.71)、鲁棒 CAS(4.72-4.73)。
- 影响: `(power, bit)` 联合分配 + 端点最坏情形成为精确鲁棒分配的标准形态。

## 18. 阶段 17: WTA 与误差反馈(4.74-4.78)

- 动机: 联合分配枚举爆炸;赢者通吃简化 + 系数误差下的纠错。
- 结果: 精确 max-min 重建(4.74)、WTA 分配(4.75)、WTA 归约枚举(4.76)、误差反馈(4.77)、UCB 证书停止(4.78)。
- 影响: 大 R 场景获得可扩展的精确/带证书变体。

## 19. 阶段 18: NOMP 式在线优化(4.79-4.82)

- 动机: 大规模在线分配: 贪心 + 可验证的局部改进。
- 结果: 最小覆盖(4.79)、leximin refine 单调终止(4.80)、单交换注水可达(4.80A)、信道失配(4.81)、QoS 缩放(4.82)。
- 影响: **WTA 贪心 + 离散 Newton refine** 成为大规模主求解器(最差目标单调不减,有限步终止);无全局最优性证明,数值对照 exact-frontier 一致。

## 20. 阶段 19: 学习-优化耦合(4.83-4.90)

- 动机: 静态分配之外的在线决策(模式选择、温度、课程)能否学习。
- 结果: MAPPO-NOMP 分解与适配器(4.83-4.84)、UCB 模式选择(4.85)、优先级中间件(4.86)、终值奖励塑形(4.87)、多温度集成(4.88)、难度课程(4.89)、UCB 温度(4.90);MAPPO 本身逊于精确联合 6.4-7.1pp。
- 影响: 学习层作为"适配器"而非"求解器"定位明确;奖励/模式由 NOMP 精确值塑形。

## 21. 阶段 20: 上界与估计器强化(2026-08,README 记录,FORMAL_PROOFS 待补)

- 动机: R>8 时擦除期望只能抽样;需要无偏低方差估计 + 确定上界。
- 结果: **Rao-Blackwell 分层 MC**(计数律精确泊松-二项,方差 ≤ 普通 MC,实测低约 39-40%);**计数条件上界**(逐点不松于无擦除上界,平均收紧约 20%);泊松-二项初等对称和 DP 的精确条件抽样;修复 ES 幂键抽样 40σ 偏置。
- 影响: 鲁棒联合分配的期望内核从"粗抽样"升级为"无偏分层 + 可证明上界";`communication_pd_with_upper_bound` 同时返回估计与三级上界。

## 22. 阶段 21: 检测信息与主动检测主线(2026-08,advice/001.md 驱动,Gate A/B)

- 动机: advice/001.md 的"任务驱动、闭环、主动获取信息"升级;用 post-communication 检测信息统一量化/链路/擦除自由度,并按两个生死门决策:Gate A(检测信息度量的**预测价值**)与 Gate B(**主动 vs 静态**)。
- 理论: 链路检测信息 `I+ = KL(p1||p0)`、`I- = KL(p0||p1)` 即 LLR 在 H1/H0 下的精确漂移(Wald);数据链 DP 收缩 `I_post <= I_quant <= I_sensing`(量化=BSC=擦除均为信道;尾质量无取消计算 + 1e-300 下限,数值严格);可检测擦除**线性缩放** `I+ = s * KL(p1_rec||p0_rec)`(擦除符号 LLR 贡献 0,构造性精确);Chernoff `C = max_s -log sum_y p1(y)^s p0(y)^(1-s) <= min(I+, I-)`;顺序检测 `P_D(n)` 由 LLR PMF 的 FFT 卷积**精确**计算(统一网格 + 重定心,无高斯近似),异构观测序列(不同报告/功率)按序卷积;NP 阈值 = H0 上尾 ≤ α 的最小网格点。
- 实现: `uav_otfs_isac/detection_information.py`;Gate A `scripts/run_detection_information_gate.py`;Gate B `scripts/run_active_detection_gate.py`;测试 `tests/test_detection_information.py`(15 项,含暴力枚举逐点对照)。
- 结果 Gate A(400 随机链路,Spearman 对 `P_D(4)/P_D(8)/n*`): **Chernoff 0.987/0.998/0.994、KL 0.978/0.976/0.975** 全面领先;单步 `ΔP_D(1)` 0.85,偏转代理 0.69-0.74(印证 G1-A 的 0.588),SNR 0.60-0.66;DP 收缩在全部 400 实例成立;16% 链路 16 周期内达不到 `P_D*=0.9`(如实报告)。**Gate A 通过**。
- 结果 Gate B(2 目标 × 3 报告 × 2 功率,预算 9,40 seeds): 预算松弛时三策略重合(无预算竞争);预算紧张时 **active**(按 `tau_pred = (eta(n+1) - n*I+)/I+` 定目标优先级)最差目标平均 T **2.08** vs static 3.10 / myopic 2.33,终态最差 `P_D` **0.599** vs 0.234/0.564。**Gate B 通过**。
- 影响: 检测信息成为新的统一主线(量化/链路/擦除/顺序检测共享同一度量);主动策略在"最差目标"指标上系统性占优(直接对应 advice 的 `min max_q E[T_q]` 目标);诚实记录预算松弛无差异与不可达链路比例。

## 23. 阶段 22: 检测感知量化与信息梯度(2026-08,advice/001.md 驱动,Gate C)

- 动机: advice 主线剩余两环 —— 量化设计自由度(感知 span/比特数)与信息梯度分配(`ΔI+/bit` 注水);需检验"KL 最大化的量化"与"信息梯度分配"是否真的逼近检测最优。
- 理论: **信息-比特单调性可证**(均匀细化量化器族 `rec(b)` 是 `rec(b+1)` 丢 LSB 的确定性函数 → DP 链 `I+(b) <= I+(b+1)`;flip/success 单调经 4.59 级联恒等式 + DP);**凹性证明性否定**(b=3 处边际跳升,不宣称注水精确最优,贪心 vs 暴力 ≤ 4.7%);**1-bit LLR 结构**(`var1 > var0` → LLR 凸 → `{LLR>t}` 为双侧窗口;`var1 < var0` → 单区间;等方差退化,解析分类 + 数值一致)。
- 实现: `uav_otfs_isac/detection_quantization.py`(`link_information_vs_bits`、`verify_*`、`information_waterfilling`(sum/max-min)、`one_bit_kl_scan`、`llr_1bit_structure`);Gate C `scripts/run_detection_quantization_gate.py`(Part A span 度量正确性 / Part B 1-bit 结构 / Part C 分配);测试 `tests/test_detection_quantization.py`(11 项,含暴力枚举对照)。
- 结果 Gate C(48 随机链路 × 11 span;20 链路 1-bit 扫描;12 实例 × 0-5 bits/报告全面枚举): **设计度量层级实证** —— `argmax I+` 的 span 在 62.5% 实例使 `P_D(4)` 低于最优 >1%(均值漂移跨设计误导),Chernoff 仅 14.6% 失败 → 层级 `精确 P_D > Chernoff > I+`;**信息梯度分配否定** —— `sum-I+`/`maxmin-I+` 注水最差目标 `P_D(4)` 0.463/0.507 vs 精确 max-min 0.794(100% 实例占优,跨设计排名失效),而 **Chernoff 最优分配 0.772 且 67% 实例与精确分配一致**(Chernoff 为正确代理);窗口对单阈值增益均值 0.00017(物理区间可忽略,保留单阈值)。**Gate C 通过(按诚实断言)**: `I+ 失败率 > 0.3 且 Chernoff 失败率 < 0.25 且 精确分配 100% 占优 且 Chernoff 分配 ≥50% 实例一致`。
- 影响: 设计自由度(span/比特)不得用 KL 均值漂移代理优化 —— advice 的"KL 最大化量化"在该设计族内是**回归**;Chernoff 是正确代理、精确 `P_D` 是基准;信息梯度注水保留为启发式并公开 4.7% 界。
- 审计修正(2026-08-16): 枚举参考从 0-3 bits 修正为 0-5 bits(与注水配置一致,原上限偏袒注水方,精确最优 0.768 → 0.794);`llr_1bit_structure` 二次方程常数项符号错误已修正(根残差 0.585-1.585 → 1e-9);`one_bit_kl_scan` 补充单区间族(`var1<var0` 的正则区域);`llr_pmf` 对单侧零质量原子从"静默置零"改为显式拒绝(原行为会破坏无限 LLR);`optimal_span` 默认度量改为 Chernoff(原默认 I+ 与 gate 结论矛盾)。

## 23A. 阶段 22 深化: 精确 max-min P_D 分配(floor-cover 定理,2026-08-16)

- 动机: 分配层原只有"无结构的边际贪心注水"(I+ 无凹性,4.7% 界,小规模暴力枚举不可扩展);需要**精确、可扩展、无假设**的分配解。
- 理论: **floor-cover 定理(4.91)**: `max min_t max_o f_{t,o}(b)` = 最小成本覆盖 —— 候选水平 `L` 的目标最小成本 `c_t(L) = min_o min{b: f ≥ L}`,最优值 = 满足 `Σ c_t(L) ≤ B` 的最大 `L`;上界(每个目标达成 L 至少花 c_t(L))与下界(按 c_t(L) 分配即达成)双向夹逼,**无凹性/单调性假设**,多项式复杂度。**NP 可容许性(4.92)**: 细化链下更细字母表的 LLR 检验最有力 ⇒ 真实 `P_D(n)` 比特单调;网格统计量非精确 LLR,实测违反 ≤ 0.008,诊断如实报告。
- 实现: `maxmin_pd_allocation`(metric="pd"/"chernoff"/"i_plus",逐选项曲线预计算 + 候选水平扫描 + 余量再分配);`option_metric_vs_bits`;`verify_pd_bits_monotonicity`(诊断)。
- 结果: Gate C 中 floor-cover 与小规模穷举**100% 一致**(最差目标 `P_D(4)` 0.794);I+ 注水 0.463/0.507 从不一致;Chernoff 代理分配 0.772(67% 实例与精确一致)保持为大规模场景的廉价代理。
- 影响: 分配层从"启发式 + 实测界"升级为"精确定理 + 可选代理";与 `tau_pred` 的逐周期闭环列为后续(§27.6)。

## 24. 阶段 23: Belief-Bellman 主动控制(Gate D1, 2026-08, advice/002-003.md 驱动)

- 动机: advice/003 建议把系统从"算法模块串联"升级为 **belief-driven closed-loop controller**(Bellman/belief/value-of-information 进入核心,ReAct/Reflexion/RAG 不进入实时感知核心),并明确决策规则: 精确预算状态 Bellman 相对最优 myopic 若无 ≥5% 实质增益,则 belief-state Actor-Critic 蒸馏(advice §1-2)不做。
- 理论: **显式预算状态 Bellman** `V_t(pi, B)`(`budget_bellman_value`,状态 = 后验 log-odds × 剩余预算;观测扣减预算、预算耗尽强制停止;宽松预算时与无约束网格值逐点一致 ≤1e-9,预算单调 `V_h(l, b+1) <= V_h(l, b)`);**Blackwell 三级剪枝前两级** —— 一级 exact dominance(LP 可行性)、二级 `value_bound_prune` 精确动作消去(续估值整条网格从未优于终止代价即当步剪除,保守化同预算层保证不误剪,与无剪枝值函数逐位一致,零信息高代价内核全步剪除;三级 approximate search 未做);**残差自适应(Reflexion 数学化,advice §4)** —— 每次观测结算 Bellman 残差 `r_t = c(a_t) + V(l_{t+1}) - V(l_t)`,对 H0/H1 两个模型条件分布标准化 `z_H = (r - mu_H)/sigma_H`,累积均值统计量 `tau = min(|mean_0|,|mean_1|)`: 单假设流下正确模型 `tau -> 0`,失配时两条件均值同时偏移、`tau` 增长;`tau > margin` 触发 robust(一步前瞻)→ explore(最大 `I+/cost`)模式,`tau < margin/2` 迟滞回归。
- 实现: `uav_otfs_isac/active_detection_bellman.py`;Gate D1 `scripts/run_bellman_detection_gate.py`(part_d1b 预算扫描 B=2..8、part_d1c 失配);测试 `tests/test_active_detection_bellman.py`。
- 结果 Gate D1b(Q=1、R=3、b=1..4、P=2 档、H=B=6、c10=c01=20;24 动作经 Blackwell 剪枝至 11): 精确 Bellman 相对最优 myopic(dpD,总代价 5.062)仅 **+0.4%**,预算扫描 B=2..8 全部 ≤2.7%(B 越大增益越大,0→2.7%),**未达 5% 实质性门槛 → 单目标尺度 "Agent 化" 不值得做,不进入 value 蒸馏第二步**;Bellman/dpD 相对信息型 myopic(τ_pred/Chernoff/c,6.2-6.5)稳定优势 15-20% —— 瓶颈在信息型调度层而非多步前瞻。
- 结果 Gate D1c(失配检测,诚实): 正确模型 τ=0.038 不触发;success 0.95→0.5 擦除失配 τ=0.666 触发并切 robust/explore;flip-only 失配不触发(统计量局限: 翻转不改 LLR 原子幅度,如实报告);适配后实现代价 ≤ naive Bellman。
- 结果 **分配-时间闭环(负结果,诚实)**: Gate B 逐周期预算循环中,预算 Bellman G 值调度(`G = V(l,b) - [c + E V(l+llr,b-c)]`,每周期按剩余预算 + 当前后验重分配)最差目标 T **3.50 vs active(τ_pred)2.08(增益 -40.7%)** —— 价值函数的停止经济学(代价目标)与 P_D 阈值目标不一致,价值代理(I+ 漂移)高估进度;该负结果直接成为阶段 24 P0 目标校正的诊断输入。
- 影响: 按决策规则 Gate D1 负结论 → **belief-state Actor-Critic 蒸馏暂缓**;确认 advice §14 路线("not 删除 NOMP/MAPPO,而是重赋理论角色");澄清必要条件 —— 价值函数必须直接嵌入 P_D 阈值目标(阶段 24 由此建立)。

## 25. 阶段 24: 目标对齐的序贯检测(Gate D2, 2026-08, advice/004.md 驱动)

- 动机: P0 诊断 —— 旧成本 Bellman 最小化"采样成本 + 贝叶斯误差",与 Gate B 的 `min_Pi max_q E_1[T_q]` s.t. `P_FA <= alpha`, `P_MD <= beta` 目标错配(即阶段 23 闭环负结果的根因);按 advice Case 1/2/3 分支逐尺度判定规划价值。
- 理论: **目标对齐延迟 Bellman**(`delay_value_iteration`): 续观分支每周期恰付 1(检测周期),声明分支由对偶价格 `(xi, zeta)` 定价,`V_h(l,b) = min{ zeta*pi, xi*(1-pi), 1 + min_{a: c(a)<=b}[ lam*c(a) + E V_{h-1}(l+llr, b-c(a)) ] }`;**数值校准双阈值停止**(P1,`calibrate_sprt_boundaries` 围绕 Wald 值网格扫描 + MC;离散量化+BSC+擦除核下 Wald 近似不精确 —— 预算紧时 Wald 边界违反约束,校准 A*≈2.34/B*≈-3.54 等更快且达标),`T_q` 从此是严格 stopping time 替代 P_D(n) checkpoint;**ν 加权 min-max**(`max_q E[T_q] = max_nu sum_q nu_q E[T_q]`,每周期续观成本 nu_q);**约束内嵌价值函数**(`joint_delay_value`,状态 `(l1,l2,B)`,停止边界外 terminal 0、带内 1e9 —— 消除价格-边界错配,即早期 8.31 反超 myopic 的根因)。
- 实现: `active_detection_bellman.py`(`delay_value_iteration`、`calibrate_sprt_boundaries`、`joint_delay_value`、`make_deployable_controllers`);Gate D2 `scripts/run_d2_objective_gate.py`(part_d2a/part_d2b)、`scripts/run_d2_deployment_gate.py`;测试 `tests/test_active_detection_bellman.py`。
- 结果 D2-A(Q=1、R=3、b=1..4、P=2、H=10、B=16、α=β=0.05): 目标对齐后 delay-Bellman `E_1[T]=3.85` 相对 one-step exact-P_D 动作选择 2.43 慢 58%,相对信息型 myopic(4.59)快 16% —— **单目标无 long-horizon 规划价值**(advice Case 1 方向,如实);旧成本 Bellman 2.05 但 P_FA=0.102/P_MD=0.118 违反约束(目标错配的代价)。
- 结果 D2-B(Q=2 strong 10/16-flip0.02-succ0.98 + weak 7/11-flip0.08-succ0.9,共享预算 30,min-max via ν 扫描): 精确联合序贯 oracle(`joint_delay_value`)最差目标 H1 延迟 **5.57 vs myopic ΔP_D 6.48(+14.1%)vs static floor-cover 11.39 vs τ_pred 16.98**,误差 (0.033,0.077) vs (0.045,0.085) 相当或略优,**超过 5% 实质性门槛 → Case 2 正结果:多目标竞争创造规划价值**;weak 目标 P_MD 略超 0.05 为共享预算资源限制(所有策略同等),如实报告。方法教训: 约束必须内嵌动态规划而非后置覆盖。
- 结果 D2-D(可部署控制器,advice Case 3): 精确 oracle 状态空间指数级不可部署,转向 O(Q) 线性家族(`make_deployable_controllers`: dual G 值 `argmax_{q,a}[nu_q(V_q - (c + E V_q)) - lam c]` / Whittle 指数 / 一-两步 rollout,共用每目标延迟值 + 校准边界): **预算充足(45)时 Q=2 dual 4.39 vs oracle 4.23(部署 gap 仅 +3.9%,捕获 96% 规划价值)vs myopic 4.93(+10.8%);Q=3 rollout_1step 6.81 vs myopic 7.43(+8.4%,规划价值扩展到三目标)**;每周期决策 2.5-6.5ms、单目标值构建 2-4s,实时可部署。资源不足区(B=30 三目标)差距收敛,如实报告。结论: dual G 值/rollout 为推荐部署方案,oracle 为性能上界。
- 影响: 目标对齐序贯检测成为新理论主线 —— **多目标资源竞争创造规划价值**(单目标不创造,边界如实);约束内嵌 DP 为方法学教训;τ_pred 降级为基线;oracle-可部署 gap 量化(≈4%)使理论可落地。

## 25. 阶段 24 深化: 系统级纠偏与分布式信息审计(2026-08-17,advice/005.md 驱动,Gate F0)

- 动机: advice/005 做系统级纠偏 —— 停止 Exact-first / Fusion-center-first / Algorithm-first,正式定位为"**通信受限条件下的分布式多 UAV 任务驱动协同检测**";不再增加第 25 个集中式理论阶段,而是重建"谁是主线、谁是工具"的层级;第一步是修改 `SYSTEM_MODEL` 系统边界,第二步立即做 **F0 Distributed Information Audit**。
- 理论: **信息结构隔离审计** —— 决策规则固定(局部 dual G 值 `J_{i,q,a} = ν_q[V(l,b) − (cycle + c + E V(l+llr,b−c))] − λc` + 邻居 intent 拥塞价格 `ψ = −η·count`),仅变化信息结构: A centralized(全局 belief,oracle)/ B full_message(精确证据 token,42 bits/广播)/ C compact_token(19-bit token,LLR 5 bits 量化)/ D local_only(零通信);每目标固定 owner,owner belief 按逐模式校准的双阈值停止;**逐模式校准(方法学)**: C 的 belief 在通信域维护,阈值在量化核上校准(精确核阈值会被中值量化夸大漂移);**差距分解** `Δ_decentral = J_B − J_A`、`Δ_comm = J_C − J_B`、`Δ_coop = J_D − J_C`(advice §13/§15);**稳定性指标**(advice §18): conflict 率、duplicate 率、role switch 率、belief 分歧 D_L;token bits 可行性边界(infeasible region,advice §20 C4)。
- 实现: `uav_otfs_isac/distributed_audit.py`(场景构建、逐模式校准、通信域量化核、四系统 MC 模拟、审计汇总);Gate F0 `scripts/run_distributed_audit_gate.py`(含 token 可行性扫描 4/5/6 bits 与校准种子敏感性扫描);测试 `tests/test_distributed_audit.py`(11 项: 场景确定性/量化界/核不变性/四模式指标范围/信息排序结构/4-bit 不可行/稳定校准确定性/审计冒烟)。
- 结果 Gate F0(K=6,Q=3,随机链路 + 弱目标,4 seeds × 800 runs,两阶段稳定校准(scan 300 + verify 2000,固定种子),`results/distributed_audit_gate.json`): 最差目标 H1 延迟 **A 4.49 / B 5.26 / C 5.36 / D 32.53**;Δ_decentral = +0.77(+17.1%,实质存在 → ownership/冲突消解/分布式价格为研究重点);Δ_comm = +0.11(+2.0%,校准种子敏感性 +1.8%…+6.3%,方向 C ≥ B 与信息论一致,量化损失小但非零);Δ_coop = +27.2(+507%)且 **D 无法满足错误约束**(P_MD 0.48 > β,40 周期截尾)→ **协同是必要的**;token 可行性: 4-bit 不可行、5/6-bit 可行(infeasible region 实证);稳定性: conflict 0.43–0.69、duplicate ~1.0(min-max 瓶颈聚焦)、role switch 0、D_L(B/C ~0.75–0.78,D 3.44)。**Gate F0 通过**(ordering_holds + 运行系统 A/B/C 错误约束满足)。计算性能: 单 UAV 决策 ~0.42 ms、单目标值构建 ~0.6 s、K=8/Q=4 结构保持(A 3.92 < B 4.57 ≈ C 4.65 < D 35.78)。
- 方法学修正(校准稳定性): 单级 MC 校准(300 runs)在近并列可行 (A,B) 间翻转,曾使 Δ_comm 为负(伪影);两阶段稳定校准消除该伪影,并把残余校准种子敏感性写入 JSON(3 个校准种子,Δ_comm +0.09…+0.35 周期)。
- 影响: 分布式信息结构成为系统模型的一等公民(`I_{i,t}`、局部 belief、token、P-DIST、节点级预算);Exact/联合 oracle 降为离线审计,dual G 值/rollout 保留为局部动作价值层,NOMP/MAPPO 进入候选箱;下一步由规模审计(阶段 26)结果决定,不再预先承诺方向。

## 26. 阶段 25 深化: 规模审计(2026-08-17,advice/006 驱动,Gate F0-S)

- 动机: 收敛 —— 不再扩大系统模型(FOV/安全距离/动态 owner/coalition/mobility/拓扑全部冻结为假设与后续扩展),论文主线锁死为"通信受限分布式协同检测",唯一主问题: **紧凑证据交换和局部任务决策能否在 UAV/目标规模增加时维持可靠检测?** 全部机制冻结,唯一变化 (K,Q) ∈ {(6,3),(8,4),(12,6),(16,8)}(K/Q=2)。
- 理论/实现: 冻结协议复用 `distributed_audit`(两阶段稳定校准、19-bit token、dual-G + ψ、固定 owner、full mesh);新 gate `scripts/run_distributed_audit_scaling.py`(五个指标 + 三个预设 Gate + first-bottleneck 判定);测试 `tests/test_distributed_audit_scaling.py`(4 项)。
- 结果(4 seeds × 400 runs,~47 min,`results/distributed_audit_scaling.json`): 主线 J(C) **5.37 / 3.97 / 5.32 / 6.08**(±0.14-0.20);P_MD^max 0.061/0.042/0.070/0.040,P_FA=0;tx/UAV 恒 19 bits,**rx/UAV = 19×(K−1) 线性增长(95→285)**;T_decision/UAV 0.50→0.89 ms。**Gate A 未通过**(J +13.1% > 10%,P_MD 在 (12,6) 0.070 略超 β+2pp)→ **第一个瓶颈:检测层延迟增长(最大档),下一步研究 target allocation / resource competition**;**Gate B 通过**(决策成本亚线性于 Q、与 K 无关);**Gate C 通过**(tx 恒定,rx 线性 —— 接收侧拓扑是通信扩展的结构性成本,非当前约束);零通信基线 (16,8) 完全失效(P_MD 1.0)。
- 方法学: J 非单调((8,4) 最优)部分来自弱目标占比随 Q 下降(仅 q=0 为弱,冻结场景生成);每档单一场景抽样,J 的 ± 为模拟种子 SE;下一步方向由结果决定,不再预先承诺 F1/F2/稀疏/owner。
- 影响: **研究路线改为 "Current F0 → Scaling Audit → 根据结果再决定"**;F1/F2 从待办降为"由规模审计结果决定后才启动"(target allocation / resource competition 优先;稀疏 U2U 仅在 rx 负载先成为约束时)。

## 27. 阶段 26 深化: 目标竞争审计与分配修复(2026-08-17,advice/007 驱动,Gate F0-A)

- 动机: F0-S 判定瓶颈在 distributed task decision / allocation 层,但"分配问题"包含多种机制(starvation / over-concentration / 尺度失衡 / owner 低效 / 每目标资源自然下降);F0-A **不改任何机制**,只加 5 个逐周期诊断量,判定三种结论。
- 实现: `uav_otfs_isac/competition_audit.py`(服务率 r_q、最大空闲 H_q^idle、并发数 n_q 统计、紧迫度-分配 Spearman ρ_alloc、逐周期 regret/扭曲选择率,含 J 尺度诊断);`scripts/run_competition_audit_gate.py`(判定 Case 1/2/3);`scripts/run_allocation_fix_gate.py`(归一化索引 + 单标量扫描);`choose_actions` 增加冻结默认的可选参数(psi_gamma、eta_A、ages、normalize_gains);测试 `tests/test_competition_audit.py`(5 项)。
- 结果 F0-A(4 seeds × 400 runs,~11 min): r_min 0.417→0.232(−44%)、H_idle 4.0→5.3(+31%)、n_max=K(全集中)、ρ_alloc 0.442→0.168(−62%)、distort 0.05%→0.17% → **Case 2+3(starvation + over-concentration)**。
- 关键发现(理论+数据): **单位失配** —— 带内 dual-G 增益 1e5–1e9(停止带 1e9 终端),O(1) 加法价格/age 数学上不可比(distort≈0.1% 证明 ψ 几乎从不改变选择);**easy-target bias** —— 带内增益 ≈ 一步穿越概率×1e9,弱目标增益 −1.0 vs 易目标 +2.5e7,索引按"立即可解决性"排序,系统性低估难目标(min-max 语义丢失,ρ_alloc 下降根因)。
- 结果修复(一次一个标量,~46 min): 原样 age/ψ 曲率(γ=1.5/2)**无效**(单位失配实证);**归一化索引**后:(12,6) 价格 η=1 → **J 5.56→4.94(−11%,P_MD 0.067 达标)**;(16,8) 全配置无效(J≈5.7)→ 最大档 worst-target 由**单个内在困难目标**决定(16/8 的 q7 e1=5.68 独大,12/6 的弱目标 q0=5.61)—— 目标难度分布主导,非分配可修复(advice §9)。
- 理论: 并发观测边际价值 ∝ n⁻²(延迟 ~ A/(n·I⁺),第 n 观测者边际 ~ (A/I⁺)n⁻²)⇒ 凸价格(γ≥2)正确但须先解决单位;age = 任务级 AoI(检测 AoI,token t_stamp 语义);修复全程在 19-bit token 协议内,感知能量机会成本与信息下界 `E_1[ΣI⁺] ≥ d(1−β‖α)` 不受影响。
- 影响: 路线收敛为 **F0 → F0-S → F0-A 全部完成**;下一阶段编号由 F0-A 结果决定 —— 候选: 难度感知权重(ν 按目标难度)、Q/K 负载可行区域、或目标难度分布下的 worst-target 结构分析;不再预先命名 F1/F2。

## 28. 阶段 27 深化: 目标难度分解(2026-08-17,advice/008 驱动,Gate F0-D)

- 动机: F0-A 后唯一问题是"16/8 的 worst-target 恶化还有多少可优化分布式损失"—— J = max_q E₁[T_q] 天然有 max-over-Q 极端样本效应,"目标难"不等于"不可优化";F0-D 纯诊断(冻结含归一化协调的修正主线),不做任何优化。
- 实现: `uav_otfs_isac/difficulty_decomposition.py`(isolated_scenario 同 realization 单目标化、三层分解 run_decomposition、难度指纹 I+/Chernoff/N_useful、信息论下界 T_LB = d(1−β‖α)/Ī_max、Case A/B/C 判定);`scripts/run_difficulty_decomposition_gate.py`;`simulate_system` 增加可选归一化参数(默认冻结行为不变);测试 `tests/test_difficulty_decomposition.py`(5 项)。
- 结果(300 runs × 3 seeds,(6,3)/(12,6)/(16,8),~15 min): 最难目标分解 —— (6,3) q1: iso 52.5% / comp 61.8% / dec −14.3%(分布式优于集中,如实);(12,6) q0: 36.5% / 59.7% / 3.8% → **Case B**;(16,8) q0: 29.3% / 32.4% / **38.3%** → **Case C**。{J^iso} 中位数 1.99→1.22→1.13、最大 2.10→1.80→1.75(**无极端样本效应**);dec gap 成分: (16,8) 弱目标量化 22.3% > 投递/局部 16.0%(**证据保真度主导**,非分配);T_LB 3.79 周期/观测 → 周期下界 ≈0.24 ≪ J^iso 1.75(**远未到信息物理极限**);指纹 I+_max 0.700 / Chernoff 0.130 / N_useful 16/16。
- 影响: 最大档增长的三层归因成立(≈1/3 内在、1/3 竞争、1/3 分布式,其中量化损失最大单项);F0-A"分配项修不动 16/8"被解释为**证据保真度主导**;下一步数据驱动候选: ① Q/K 负载可行性区域(竞争份额),② **弱目标保真度自适应 token(非均匀/压扩量化)**(量化份额 22.3% 的唯一有据方向);难度感知权重被明确反对(resource-sink 风险,须 J^iso 先区分可改进性)。

## 29. 阶段 28 修正: Token 保真度审计与分解方法学修正(2026-08-17,Gate F0-E)

- 动机: F0-D 将 dec gap 的 22.3% 归因于弱目标量化 —— F0-E 用三个有理论支撑的 token 设计(同一 19-bit 预算)检验该杠杆,并发现 F0-D 本身的组成混淆。
- 实现: `distributed_audit` 增加统一编码器抽象(`uniform_quantizer`/`lloyd_max_quantizer`(质心条件、H1 加权)/`mu_law_quantizer`(压扩近零细分)/`build_token_quantizer`(按目标等权)/`quantize_with`),`quantized_kernels`/`calibrate_target_bounds`/`simulate_*` 全部支持可注入码本(默认冻结行为不变);`scripts/run_token_fidelity_gate.py`(配对固定阈值 + 重校准);`difficulty_decomposition.run_decomposition` 修正 B/cent 参考为归一化索引(cent 无价格 η=0、B 保部署价格);测试 `tests/test_token_fidelity.py`(8 项)+ `test_difficulty_decomposition.py`(5 项,含分解一致性)。
- 结果(负结果 + 方法学修正): 配对固定阈值下码本重设计 ≤±0.1 周期;**L̂ 5→10 bits(预算内重分配,1024 级近精确)在 (12,6) 重校准 −2.8%、(16,8) −0.2%** —— 量化速率不是瓶颈;修正后 (16,8) 最难目标 iso 29.3% / comp 34.5% / dec 36.2%(Case C 不变),dec 成分 **投递/局部 +40.6% > 量化 −4.4%**(中值量化漂移放大伪影)。
- 理论: 中值量化每 bin 有偏(漂移放大,解释负 quant share);率失真 R=10 已近精确而 J 不变 ⇒ 损失在传输路径(投递失败 + 意图 1 周期滞后),不在编码;F0-A 死载荷诊断(u/r/χ 从未被读取)支持预算内重分配但收益 ≈0。
- 影响: token 保真度方向关闭(负结果,量化速率非瓶颈);**16/8 剩余可优化空间在投递/协调层** —— 下一步候选: R_coord=2 第二轮意图新鲜化(协议允许未用)、预算内投递可靠性、Q/K 负载可行性区域;路线 F0→F0-S→F0-A→F0-D→F0-E 完成,下一阶段由结果决定。

## 30. 阶段 29 深化: 投递/协调审计与尺度自适应价格(2026-08-17,Gate F0-F)

- 动机: F0-E 后 dec gap 定位在投递/协调层;先分解(投递 vs 意图新鲜度 vs 局部决策),再优化主导项。
- 实现: `choose_actions` 增加 `counts_override`(新鲜计数注入);`simulate_system`/`simulate_competition_audit` 增加 `delivery_override`(完美投递诊断)与 `fresh_intents`(R_coord=2 两轮协调: 第一轮广播基准意图、第二轮证据 token,均在 19-bit/UAV/周期预算内);测试 +4 项(冻结默认保持、选项冒烟、负结果方向)。
- 结果: **dec gap 分解(16/8)**: 完美投递 −1.2 周期(投递=主导,物理层受限);**新鲜意图负结果**(full_message +2.0、主线 +2.5 —— 价格新鲜化后过度驱散);**价格尺度扫描**: (12,6) η*=1(J 4.92)、(16,8) η*=0(J 5.69);**尺度自适应价格正式(500×4)**: (12,6) η=1 J 4.935 不变,(16,8) η=0 **J 6.014→5.680(−5.6%,P_MD 0.054、r_min 0.152→0.224)**。
- 理论: 并发观测边际价值 ∝ n⁻² ⇒ 大 K 集中=高效聚焦(弱目标延迟 ∝ 1/n),价格只应惩罚超出 K/Q=2 的浪费性重复;固定尺度价格在 K 增大时把高效聚焦一并驱散 → η* → 0。标量索引参数,协议/预算/模型零改动(通信/感知原则)。
- 影响: **尺度自适应价格(小规模保价格、大规模弃价格)成为修正主线的组成部分**;剩余可优化 headroom: (a) 投递为物理层受限(需拓扑/链路预算,稀疏 U2U 仍后置),(b) Q/K 负载可行性区域(竞争份额);下一步由结果决定。

## 31. 阶段 30 深化: 操作点校准 / 多场景统计 / FRIDS(2026-08-17,advice/009 驱动,Gate F0-G1/G2/G3)

- 动机: 锐评后收敛为一条纪律链 —— 先修操作点(P_FA=0 浪费)、再做多场景统计(规模声称无基础)、最后只测一个数学上完整的新调度器(FRIDS),不再加补丁。
- 实现: `scripts/run_operating_point_gate.py`(部署策略下坐标扫描 A);`scripts/run_multiscenario_gate.py`(场景方差分解);`uav_otfs_isac/frids.py`(ν·g 索引、单纯形下界投影、信息可行性证书)+ `scripts/run_frids_gate.py`(Current/No-price/Proposed 对比 + 生死门);测试 `tests/test_frids.py`(6 项)。
- 结果: **G1**: (12,6) 4.94→3.37(−32%)、(16,8) 5.68→4.58(−19%),P_MD 0.036-0.042;阈值保守 = 最大单一可修复损失。**G2**: 场景方差占 69-95%,16/8 平均增长仅 +3.9%(单场景 +13% 偏差)。**G3(FRIDS)**: 生死门**通过** —— (12,6) 5.16→2.75(−46.7%)、(16,8) 5.36→3.26(−39.3%),P_MD ≤0.066、P_FA≈0、5/5 场景全胜、无小规模回退。
- 调试教训(数理): FRIDS 流弱于校准流 → P_MD 超 β 由 H1 下穿 B 的 H0 误判主导,正确杠杆是**降 B**(不损 P_FA);每场景 δ_B 匹配需"扫描短名单 + 权威 MC 复核"(单层扫描 MC 噪声选边缘 δ);ν 单纯形需排除已判决目标(否则其价格残留压制其余)。
- 影响: **FRIDS 成为修正主线**(检测缺陷对偶、可靠信息调度、可行性感知公平、规模无关协调四项创新);η(K) 两点拟合退役;"规模下维持检测"从"答不能"转为"大幅改善但 16/8 仍高于 12/6(+19%)";剩余: FRIDS 全规模曲线、可行性区域、更多 K 档。

## 32. 阶段 31 深化: FRIDS 一致性审计与 FRIDS-v2(2026-08-17,advice/010 驱动,Gate F0-G4)

- 动机: FRIDS 已从"经验优秀"升级到"数学自洽 + 严格分布式 + 通信记账正确 + 可证明负载感知"的必要关口;先审计正确性(五查),再只测一个升级(mirror descent + 需求归一化)。
- 实现: `frids.py` 新增 `load_cut`(信息负载 cut)与 `simulate_frids_v2`(严格局部: 每 UAV 自己的 D^{(i)}/S^{(i)}/y^{(i)};需求归一化索引 `J=y·g/(D+ε)`;指数梯度 mirror descent);`token_bits(Q)` 规模感知(⌈log2Q⌉,死载荷 u/r 腾出,≤19 bits);`scripts/run_frids_v2_gate.py`(v1-vs-v2 四档 × 5-10 场景,自适应策略匹配 B);测试 +6 项(可靠信息恒等式、token 记账、v2 确定性、负载 cut 数学)。
- 结果: **审计闭合** —— g 恒等式数值精确(无重复计投递)、对偶一致性经需求归一化修复、provenance 严格局部化、token Q=8 记账修复;**FRIDS-v2 采纳**(生死门): (6,3) 3.60→3.32(−7.6%)、(8,4) 3.95→3.45(−12.7%)、(12,6) 3.90→3.60(−7.6%)、(16,8) 4.16→3.53(**−15.1%**),P_MD ≤0.053 全档零违例,胜率 0.9 —— **严格局部+对偶一致的 v2 既更理论干净又更快**(provenance 修复无性能代价)。
- 理论: 负载 cut ρ(S)>1 ⇒ 子集不可按时完成(Theorem 4 必要条件);实测 ρ_full ≈0.03-0.05 ≪ 1(该场景族信息容量不绑定,此前"竞争份额"是分配动力学,已被 FRIDS 消化);mirror descent 静态松弛 regret O(√(T log Q)) 留作形式化。
- 影响: FRIDS-v2 成为当前主线(检测缺陷对偶 + 可靠信息调度 + 可行性感知公平 + 规模无关协调 + **严格局部性**);剩余: 可行性区域实证(极端 Q/K、低可靠区)、定理 1-4 形式化(FORMAL_PROOFS)、v2 全场景 10-seed 复核。

## 33. 阶段 32 深化: FRIDS 理论形式化与可靠性鲁棒性(2026-08-17,advice/011 驱动,Gate F0-G5)

- 动机: FRIDS-v2 实验已领先理论 —— 先形式化四定理(可靠信息恒等式/原始-对偶/mirror-descent 界/负载 cut),再只测一个鲁棒性 Gate(U2U 可靠性估计错误的健壮性)。
- 实现: `FORMAL_PROOFS.md` §5B 定理 4.94-4.97(含 traceability 行与非主张);`frids.simulate_frids_v2` 增加 `delivery_matrix`(真实投递)与 `s_for_g`(假定可靠性)分离;`scripts/run_reliability_robustness_gate.py`(κ_s ∈ {1.0,0.9,0.8,0.6} 全局 + 逐链路探针,nominal/robust/oracle 三变体);测试 +1 项(尺度不变性)。
- 结果(负结果 + 洞察): **nominal ≡ robust(robust_gain ≡ 0.0)** —— 均匀 (1−ε) 缩放不改变 argmax,鲁棒变体 NO-OP;全局 κ 失配仅 +2%(s=1 clip 的非均匀失真)、逐链路失配 +1.7%(mirror descent 价格动力学吸收)→ 按生死门**保持 FRIDS-v2,不引入 robust layer**;理论: `g=s·I+` 单调线性 ⇒ 区间最坏点封闭,但鲁棒必要性由尺度不变性消解。
- 影响: 算法部分开始收口 —— FRIDS-v2(严格局部、对偶一致、量纲闭合、可靠性失配稳健)+ 四定理理论主干;剩余投稿工作转向 **Q/K × U2U reliability 可行性边界与通信预算区域**(通信定价 FRIDS 仅在预算真正成为 binding constraint 后)、10-seed 复核、可行性区域实证。

## 34. 阶段 33 深化: 可行性包络与瓶颈子集定律(2026-08-17,advice/012 驱动,Gate F0-G6)

- 动机: FRIDS-v2 冻结;研究转向"什么负载/通信条件下必然可行/不可行"—— 把负载 cut 升级为最强 cut ρ*(瓶颈子集定律),主动把系统推向边界。
- 实现: `uav_otfs_isac/feasibility.py`(Fujishige-Wolfe 最小范数点子模最小化 + 二分求 ρ*,贪心基多胞形线性预言机,SLSQP 内层 QP;与暴力枚举 Q≤8 逐点一致);`scripts/run_feasibility_envelope_gate.py`(K=16,Q/K∈{0.25..2} × s∈{0.95..0.2} 30 格点相图 + Γ + 三类不可行分离);基础修复: Q>K 的 owner 循环分配(q%K)、token 记账级联丢弃死载荷(u→r→χ→stamp,⌈log2Q⌉ 恒 ≤19 bits);测试 `tests/test_feasibility.py`(7 项)。
- 结果: **27 Green / 3 Yellow / 0 Red**;`ρ_I* ≤ 0.11`(信息容量从不绑定,即使 Q/K=2、s=0.2)、`ρ_C = 0.71 < 1`(通信预算不绑定)⇒ 非 sensing-limited、非 communication-limited;3 个 Yellow(Q16/s0.6、Q32/s0.95、Q32/s0.8)为 P_MD 0.071-0.083 的**边缘 coordination gap**(≈1pp);**Γ∈[0.75,1.0] ⇒ FRIDS 已贴近可实现边界,停止调度器优化**(advice/012 规则)。
- 理论: 定理 4.98 瓶颈子集定律(次模结构 + 多项式求解);相图把"算法失败"与"物理不可行"彻底分离 —— 当前系统相图: distributed-feasible(27)+ 边缘 coordination-limited(3),无 sensing/communication-limited 区。
- 影响: 算法部分**正式收口**(不再改 FRIDS);剩余投稿工作: 10-seed 复核、极端参数区(更小 s、更大 Q/K)使 ρ* 激活、通信预算扫描(ρ_C 随 B̄_rx 变化)、可行性区域作为 Contribution 4 写入论文。

## 35. 阶段 34 深化: 物理 airtime 报告门(2026-08-19,advice/013 驱动,Gate F0-G7)

- 动机: F0-G6 证明 ρ_I*≤0.11、ρ_C=0.71<1,调度器已冻结;013 指出真正的物理缺口是"19-bit 账本不是 waveform-derived airtime"—— 把 U2U 从 bit 计数升级为**真实 airtime/capacity 约束**,并只增加 report/no-report 唯一自由度。
- 实现: `uav_otfs_isac/airtime.py`(Shannon 容量**上界** `R=W log2(1+γ)`,γ 由 outage 成功反解 ⇒ 容量与投递同一链路统计,感知信道独立;token airtime `τ=b_tok/R` 秒;全网格接收负载 `L_i=Σ z_j τ_ji`,超载 `min(1,T_air/L_i)` 队溢出丢弃;report/no-report 门 `z_i`,价格 `λ=λ_base+λ_dual`(任务机会成本基线 + 本地负载对偶上升,冷启动用本地稀缺度 `ρ_est=Στ/T_air` 立即生效));两种取值 `deficit`(严格联合-LP 对偶 `y·g/(D+ε)−λ·c_air`,013 §2 原式)与 `info`(价格按 deficit 归一化 ⇒ D 抵消退化为比较 `y·g` 与 `λ·c_air`,Lemma 4.101);修复: 价格曾泄漏进目标选择使 always 基线随 mu_c 漂移,现目标选择全模式冻结为 FRIDS-v2 argmax;`scripts/run_airtime_reporting_gate.py`(K=16/Q=8,ρ_full∈{0.5,1.0,1.5}×五方法×两 value mode);测试 `tests/test_airtime.py`(13 项)。
- 结果(**adopted with caution,单门通过**): 非拥塞/临界区 deficit mode **airtime 削减 52.8% 而 ΔJ=−0.4%**(`Delta J≤2%` + `削减≥30%` 生命门**通过**)—— 一半 U2U 开销是低价值报告;拥塞区未过 5% 门: info mode 相对 always +2.5%、相对等量 random +3.3%(选择价值存在,且优于中心 oracle),但最坏目标延迟是**总量信息驱动**,报告级选择杠杆有限;`y*g/D` 归一化价值选择把 airtime 损失集中在高 deficit 目标(其报告归一化价值天然小),对延迟目标反最坏目标。
- 理论: `FORMAL_PROOFS.md` §5C Lemma 4.99(无 idle 动作 ⇒ 公共加法价格 NO-OP,配 no-report 才生效)、4.100(token airtime=bits/rate,Shannon 上界)、4.101(价格 deficit 归一化 ⇒ D 抵消,`y·g` vs `λ·τ`)、4.102(全网格接收负载 + 队溢出存活,投递质量被预算封顶)、4.103(负载对偶上升 + 本地稀缺度冷启动;bang-bang 极限环诚实标注)。
- 影响: airtime 价值定位在**可行性/通信效率层**而非延迟层(把不可行拥塞帧变成可优雅降级 + ~50% 通信削减);按 013 规则**不继续调 threshold**,关闭"拥塞区延迟改善"诉求;下一步按 013 优先级推进 **G8 证据相关性审计**(先 G8A 不动 FRIDS,测 `Σg_iq` 是否高估联合 `G_q(S)`,冗余比显著再考虑 conditional-KL)。

## 36. 阶段 35 深化: 证据相关性审计(2026-08-19,advice/013 驱动,Gate F0-G8A)

- 动机: FRIDS 调度层以每 UAV singleton 可靠信息 `g_iq` 加总,隐含假设观测独立;013 §5-8 指出共享杂波/目标散射的 UAV 证据不会提供两份独立信息,先做**不动 FRIDS 的纯审计**——受控公共相关系数 `rho_s`,测 `Σg_iq` 相对联合 `G_q(S)=D_KL(P1^{Y_S}||P0^{Y_S})` 是否高估,冗余比 `R_q=1-G_q(S)/Σg_iq`,显著再启动 conditional-KL。
- 实现: `uav_otfs_isac/evidence_correlation.py`(高斯公共因子模型 `Y_i=δ_i·H1+√ρ·C+√(1-ρ)·N_i`,δ_i=√(2·g_iq) 由可靠信息反解;Sherman-Morrison 闭式 `R=(ρ/(1-ρ))·(a/(1+(|S|-1)ρ)−1)`,a=对齐度;KL chain rule 条件边际 `ΔG_{i|S,q}=G(S∪{i})−G(S)≥0`;Monte-Carlo 采样验证;高斯序贯检测验证延迟后果);`scripts/run_evidence_dependence_gate.py`(K=16/Q=8,ρ∈{0,.2,.5,.8}×逐目标 R/对齐/条件边际/序贯延迟比);测试 `tests/test_evidence_correlation.py`(10 项: 闭式=采样、ρ=0 退化、单体=边际、R 随 ρ/|S| 增、条件边际非负、非次模见证、序贯延迟比)。
- 结果(**显著,启动 G8-B**): 该场景族目标内 UAV 剖面高度均匀(对齐 ~15/16),故 ρ=0.5 时最坏目标冗余比 **R=80.5%**,解析延迟低估 ~5x,序贯测试实测 top-4 联合延迟比 ~2x(含 Wald 过冲修正);`R<5%` 关门规则未触发 → **G8-B conditional-information FRIDS 有据可依**。
- 理论: `FORMAL_PROOFS.md` §5D 定理 4.104(公共因子联合 KL 闭式)、4.105(KL chain rule 条件边际 ≥0,且**不**被 singleton 上界——G 非次模,013 警示验证: 高 ρ 时条件边际可超 singleton,记录了超加性见证)、4.106(冗余的延迟后果 `T_joint/T_sing ~ 1/(1-R)`)。非主张: 同协方差模型在极端 ρ=0.8 对异质联合会因噪声白化回弯,生命门取物理区 ρ∈{0.2,0.5}。
- 影响: singleton 加总在相关观测下**严重高估**联合检测信息,FRIDS 的资源/缺陷记账存在冗余;G8-B 将把调度值从 singleton `g_iq` 替换为条件边际 `ΔG_{i|Ŝ_q,q}`(仅用 UAV i 实际收到的 intent/token 推断 `Ŝ`,仍严格满足 `a_{i,t}=π_i(I_{i,t})`)。

## 37. 阶段 36 深化: Conditional-Reliable-Information FRIDS(2026-08-19,advice/013 §7 / advice/015,Gate F0-G8B)

- 动机: G8-A 判定相关性显著(R=80.5%),启动 conditional-KL 调度;015 §4 方案: 只替换调度值 `J=y·ΔG/(D+ε)`,`ΔG_{i|S,q}=G(S∪{i})−G(S)`(KL chain rule),`S` 仅由 UAV i 实际收到的 intent 推断(严格局部);创新定位"不用 RL 学 diversity,用 conditional KL 定义 diversity 的任务价值"。
- 实现: `uav_otfs_isac/conditional_frids.py`(可靠 δ=√(2·g_reliable) 用于调度值、观测 δ=√(2·max i_plus) 用于服务记账;`rho_s` 为调度器相信的相关、`world_rho` 为世界相关,`world_rho>0` 时镜像下降服务缺口用联合 `G_q(S_received)` 替代 singleton 加总);`scripts/run_conditional_frids_gate.py`(Step1 值交换独立世界 + Step2 相关世界,3 场景配对);测试 `tests/test_conditional_frids.py`(8 项,含 **rho_s=0 ≡ FRIDS-v2 逐延迟一致**的 sanity)。
- 结果(**性能符合预期,采纳候选**): Step1 值交换 ρ_s=0.5 聚合 **J −4.2%**(场景 -9.5%/-5.4%/+1.9%,2/3 改善)—— 条件值作为**原理化防堆积信号**(冗余折扣替代已退役的 congestion price,即 F0-A n^-2 律的 KL 化);Step2 相关世界 ρ_world=0.5 下条件调度相对同世界 singleton **J −8.7%**(3/3 场景 -12.8%/-11.9%/-1.2%)—— 正是 G8-A 量化的 singleton 高估的修正;误差全在 β+2pp 内。
- 理论: `FORMAL_PROOFS.md` §5D 定理 4.107(ρ=0 时条件调度 ≡ FRIDS-v2;ρ>0 冗余折扣 = 原理化防堆积;实测改善)。
- 非主张: Step1 改善场景相关(1/3 回归 +1.9%);调度器需知 ρ_s(参数非估计);coalition 估计是 1 周期滞后+投递受限的 intent 图;相关世界只在服务记账层建模(非完整相关检测信念)。
- 影响: FRIDS 的调度值升级为 conditional reliable information,弱目标在相关世界受益最大;与 G7 airtime 门、G8-A 审计构成"可行性/相关性/条件信息"三层贡献。

## 38. 阶段 37: 深度审计(2026-08-19,advice/017)

- 动机: 先于后续优化,对 G7/G8A/G8B 做深度审计(代码、统计稳健性、模型敏感性、过拟合),修订已有文档中的数字主张。
- 实现: 删除死代码 `evidence_correlation.redundancy_audit`(含 bug: 把整个 (K,Q) 矩阵当单一 coalition);`conditional_frids` 增加 `coalition_mode="perfect"` 口径(上一周期真集合 oracle,用于量化本地 intent 时效代价);`scripts/audit_g7_robustness.py`(G7 多场景: 固定 λ_base 出样本 + 每场景重选 λ_base 按 Gate 自身选择规则)、`scripts/audit_g8_robustness.py`(G8B 5 场景配对 bootstrap CI + staleness + ρ_s 单调性;G8A 集中剖面模型敏感性)。
- 结果(**重大修订**): (1) **G7 定量主张不稳健**——非拥塞生命门 1/3 场景通过、拥塞门 0/3;固定 λ_base 在临界区 +14% 全场景退化;"52.8% 削减"是场景 0 样本内结果,不可写入论文主张;(2) **G8A 的 R 剖面敏感**——同质剖面 R=80%,但集中剖面(1 主导+15 弱)R=−7%(联合 KL 反超 singleton,噪声白化效应),"singleton 高估 80%"不可外推;(3) **G8B 仅一个统计成立主张**——相关世界下一致条件调度 vs 同世界 singleton:+6.0%(95% CI [+1.1%,+10.8%],4/5 场景);Step1 独立世界改善不复现(均值 +1.9%,CI 含 0,3/5 退化);本地 intent 时效代价仅 ~0.7%;ρ_s 响应非单调(过拟合风险)。
- 影响: 论文主张按审计修订(SYSTEM_MODEL 12.14-12.16、FORMAL_PROOFS 5C/5D);后续优化只保留"相关世界下条件调度 +6%"这一有统计支撑的方向(需 ρ_s 估计 + 8-10 场景复核),或回到 G6-R 统计收牢与论文写作。

## 39. 阶段 38: Covariance-Native Conditional Information(2026-08-19,advice/018 驱动,Gate F0-G8C)

- 动机: 深审计否定 G7 定量主张、G8A 的 R 剖面敏感;唯一有统计支撑的是相关世界下条件调度 +6%;018 提出把标量 ρ_s 淘汰为 covariance-native Schur 条件信息,先修 Theorem 4.107 表述矛盾,再做 3 profile × 10 场景生死门。
- 实现: 修正 Theorem 4.107(redundancy discount → conditional innovation effect);`uav_otfs_isac/covariance_conditional.py`(Schur 补条件信息 `ΔG=½δ_{i|S}²/v_{i|S}` + 一般高斯 KL + OTFS/DD 物理协方差源(双基地几何×Doppler 重叠×共享杂波)+ 3 profile 生成 + Gaussian-evidence FRIDS 模拟(owner 联合 LLR 融合,严格局部 coalition);修复设计 bug: 服务缺口从"全额联合 G_q(S)"改为"边际 ΔG"入账后三 profile 全部转正);`scripts/run_covariance_conditional_gate.py`(10 场景 × homogeneous/heterogeneous/concentrated × 4 方法 + 生死门);测试 +14 项(含 Schur=G(S∪{i})−G(S) 数值验证、冗余/协同案例、一般 KL 退化、独立世界恒等)。
- 结果(**REJECTED,按 018 规则收口**): covariance-native 均值增益 homogeneous +5.0%(win 7/10)、heterogeneous +2.2%(CI 含 0,win 6/10)、concentrated +4.6%(CI 含 0,win 6/10);**最坏集中剖面回退 11.7%**(违反"错误去冗余 >2% 回退"规则);连 perfect-coalition oracle 也不能稳定胜过;三 profile 均未达 win ≥ 8/10。
- 理论: Theorem 4.108(Schur 补条件信息,统一 redundancy/synergy 为 conditional innovation effect,数值验证);该公式作为**理论对象成立**,但调度收益不稳健。
- 影响: **算法彻底收口到 FRIDS-v2**(G6 冻结),correlation scheduler 主贡献关闭,不再调 covariance;G7 airtime 保留为 physical accounting 机制(非 headline);下一步按 018 收敛路线回到 **G6-R(10-seed 统计收牢)+ 论文写作/可行性边界(Contribution 4)**。

## 40. 阶段 39: Local-Dual Consistency Audit(2026-08-19,advice/020 驱动,Gate F0-G9A)

- 动机: 020 提出精炼式升级——理论 LP 有唯一公共对偶价 y,部署算法却有 K 个局部价 y^(i);回答"剩余 coordination gap 是否来自 local dual disagreement",以及"多大的局部 belief/price 误差下动作不变"。先做纯诊断,不改算法。
- 实现: `frids.simulate_frids_v2` 增加 `price_mode`(local 冻结 / common-price oracle: 动作用 `y=mean_i y^(i)`,其余全局部)与 `audit=True`(每周期追踪 D_y 局部价格分歧、D_v 归一化价值分歧、owner-局部 deficit gap、动作不变性证书 P(m_i>2E_i)(Theorem 4.109)、**实现的动作改变率并分解为价格/缺陷贡献**;默认路径逐字节不变,测试覆盖);`scripts/run_local_dual_audit.py`(K=16/Q=8,3 场景);测试 +3 项(common-price 运行与诊断、**动作不变性定理数值验证** 200 随机扰动)。
- 结果(**冻结 FRIDS-v2**): common-price oracle 最坏目标延迟仅 **−1.8%**(<2%)——局部对偶分歧**不是延迟瓶颈**;但严格证书不成立(P(m>2E)≈0,实现动作改变率 ~67%,价格 ~50% / 缺陷 ~52%,owner-局部 deficit gap ~2.11)——局部动作确实偏离公共价/owner 锚定理想,但最坏延迟是**总量信息驱动**,对"谁服务哪个目标"鲁棒。
- 理论: Theorem 4.109 分布式动作不变性证书(`|Ĵ−J| ≤ V_max·ε_y+ε_v+ε_y·ε_v`,m_i>2E_i ⇒ argmax 不变;token 年龄界 `|D^o−D̂|≤aL^max+ε_L` 进入 ε_v)。非主张: 证书是**充分非必要**条件,实测不成立,论文不得声称动作级一致,只能声称延迟级鲁棒。
- 影响: **算法正式收口**(020 路线图第一分支: gap<2% ⇒ FRIDS-v2 足够);G7/G8 全部定位为 boundary/negative;后续转 **G6-R(10-seed 统计收牢)+ 可行性边界(Contribution 4)+ 论文写作**。

## 41. 阶段 40: System Bottleneck Audit v2(2026-08-19,advice/022 驱动)

- 动机: 022 定审计驱动优化——FRIDS-v2 冻结,用 4 个隔离 oracle 找"第一大真实 headroom",谁的 gap>5% 下一轮只优化谁。问题从"FRIDS 公式还能怎么改"变为"哪些假设会导致结论失真"。
- 实现: `build_distributed_scenario` 增加 `snr_shift`(ideal-evidence oracle: 统一强感知);`simulate_frids_v2` 增加 `mobility`(冻结策略对 ±5%/周期证据随机游走的响应,默认 None 逐字节不变);`scripts/run_system_bottleneck_audit.py`(K=16/Q=8,3 场景 × 4 oracle + Δ_dual + 跨场景 median/p90/max + 配对 bootstrap CI);测试 +1 项(oracle 钩子)。
- 结果(**headroom 分解**): sensing **+18.2%**(CI [+6.4%,+30.0%]) > comm **+12.5%**(CI [+0.7%,+28.1%]) > owner **+4.3%**(CI [+0.0%,+12.8%]) > dual **+1.8%** > mobility **−0.7%**。**分布式任务层基本最优**(owner/dual/mobility 全 <5%);剩余 headroom 在**物理层**: 感知(最大)与通信。
- 影响: 按 022 规则,下一轮 one-variable repair 锁定 **sensing 层**(gap 18.2%>5%);其余全冻结;FRIDS-v2 任务层正式过关。跨场景统计: (16,8) J median 3.41 / p90 3.73 / max 3.81。

## 42. 阶段 41: Resource-Conserving Sensing Audit(2026-08-19,advice/024 驱动,Gate G10)

- 动机: 024 质疑"+4dB 18.2%"可能只是增加总功率(硬件 headroom,低价值);先把 sensing headroom 分成"可优化分配"与"硬件",并审固定 TB 的 OTFS DD 资源形状。FRIDS-v2 全冻结。
- 实现: `build_distributed_scenario` 增加 `snr_shift`/`powers`/`dd_grid`+`dd_physics`(标准 OTFS sinc² 分数多普勒/时延泄漏,固定 N_d·N_l);`simulate_frids_v2` 增加 `power_cap`(每目标感知功率档)与感知功率记账;`calibrate_target_bounds` 增加 `power_cap` 过滤(阈值匹配实际证据);`scripts/run_sensing_resource_audit.py`(G10-A 12 场景 +4dB 复现 / G10-B 能量守恒 power oracle / G10-C fixed-TB OTFS);测试 +2 项(power_cap 能量、dd_grid 泄漏改变证据)。
- 结果(**三连结论**): (1) **G10-A**: +4dB gap **+18.7%**,CI [+13.9%,+23.9%],win 1.00 → **sensing 确认为主瓶颈**(12 场景满足 10-20 要求);(2) **G10-B**: 能量守恒 oracle Δ_E=**−5.1%**(CI 含 0,噪声)→ **Case A**: 同能量重分配不帮助,+4dB 是**硬件 headroom**(额外总功率),不做 power 算法(G10-D 不启动);(3) **G10-C**: 固定 TB 下 DD 网格形状改变证据尺度 ~37%,并**决定误差工作点的可行性**(多数网格-场景使弱目标在 P_FA=P_MD=0.05 不可行)→ **OTFS 进入算法机制**,论文主张依赖 OTFS grid design(下一杠杆是 fixed-TB OTFS sensing design)。
- 影响: sensing 是主瓶颈但非能量可分配;OTFS DD 资源形状是**资源守恒**的感知杠杆(不增功率,只改网格形状);回答长期悬而未决的"固定 TB 下 OTFS-grid scaling"开放问题。FRIDS-v2 任务层继续冻结;下一步 fixed-TB OTFS DD sensing design(或 comm 12.5% 待 sensing 修复后重新分解)。

## 43. 阶段 42: Physical-to-Task Information Shaping(2026-08-19,advice/001 驱动,Gate G11)

- 动机: 001 聚拢+升华——研究点固定为 "Reliable Detection Information Shaping and Distributed Scheduling for Multi-UAV OTFS-ISAC",唯一货币 `g_{iq}=s_{io_q}I^+_{iq}(G)`;核心理论是把 OTFS 网格与瓶颈子集可行性定律连接(`H_LB(G)=max_S D(S)/F_G(S)`),并验证其能否预测最终 J(G)。
- 实现: `scripts/run_information_shaping_gate.py`(G11-A 物理账本闭合(TB=4096 固定、能量结构相同)/ G11-B 任务信息可行性定律(H_LB 跨网格预测 J 的 Spearman + ρ*>1 预测不可行)/ G11-C 盲验证(task-optimal vs SNR-optimal vs balanced vs current,设计-留出划分,calib-margin=2.0 解决弱证据可行性))。
- 结果(**混合/负,按 001 规则收口**): G11-B 可行格 Spearman(H_LB, J)=**0.67**(<0.7 门槛),ρ*>1 仅预测 10 个不可行格中的 1 个 → **H_LB 是必要条件,不是紧的延迟预测器**;G11-C 盲验证退化(task-optimal 与 SNR-optimal 都与 balanced 重合,留出无 >5% 稳定优势)。
- 影响: **保留可行性定律(必要条件;G10-C 已证网格决定工作点可行性),但停止 OTFS grid 优化作为延迟改进杠杆**;研究问题保持在 reliable-information shaping 框架;下一审计回到 **communication ~12.5% headroom**(sensing 已审完: 主瓶颈但非能量可分配、H_LB 不紧预测)。

---

## 26. 发展主线小结

| 主线 | 起点 | 终点 | 代表定理 |
| --- | --- | --- | --- |
| 精确性→可扩展性 | 2^R 精确枚举 | Pareto 剪枝 / B&B / 注水 / NOMP refine / 分层抽样 | 4.7A, 4.7D, 4.12, 4.80, 4.70A |
| 集中→分布式 | 单一融合中心 | 对等多数 + 精确泊松-二项可行性 + 逐目标切换 | 4.20, 4.43, 4.48 |
| 名义→鲁棒 | 干净信道 | 端点最坏情形精确 DP + 机会约束 + 单调定理 | 4.58, 4.70, 4.71 |
| 静态→动态 | 固定几何 | AR(1) 移动 + MMSE 预测 + 迟滞切换 | 4.26, 4.53, 4.56 |
| 静态→主动 | 固定 WTA 分配 | τ_pred 主动目标优先级 + 精确顺序检测 | I± 漂移, Chernoff, FFT 卷积 P_D(n) |
| 代理→精确 | KL/SNR 代理 | 设计度量层级(精确 P_D > Chernoff > I+) + 信息梯度注水(公开 4.7% 界) | I+(b) 单调(DP), 凹性否定, 窗口/单区间 LLR 分类 |
| 分配→精确 | 边际贪心注水(启发式) | floor-cover 精确 max-min P_D 分配(无凹性假设, 与穷举 100% 一致) | 4.91, 4.92 |
| 静态→序贯控制 | P_D(n) 检查点式顺序检测 | 预算状态 Bellman + 数值校准双阈值 stopping time + 残差自适应失配检测 | `V_t(pi,B)`, (A*,B*), τ 统计量 |
| 代价→目标对齐 | 成本 Bellman(采样+贝叶斯误差,与目标错配) | 延迟 Bellman(续观=1 周期 + 对偶价格)+ 约束内嵌 ν min-max | D2-A/B, 5.57 vs 6.48 |
| 集中 oracle→可部署 | 指数状态空间精确联合 oracle | O(Q) dual G 值 / Whittle / rollout(部署 gap +3.9%) | D2-D |
| 集中→分布式执行 | one-fusion-center 部署假设 | `a_{i,t}=π_i(I_{i,t})` + 局部 belief + U2U token + 节点级预算(F0 信息结构隔离) | Δ_decentral 17.1%, Δ_coop 507% |
| 名义→信息域校准 | 精确核阈值用于量化证据 | 通信域 belief + 量化核逐模式校准(4-bit 不可行实证) | F0 逐模式校准 |
| 冻结→规模审计 | (K,Q)=(6,3) 单点 | K/Q=2 扫描至 (16,8): 检测 J +13.1%(Gate A 未过)、决策亚线性、rx 负载线性 | F0-S, 5.37→6.08 |
| 分配→诊断 | 假定"分配算法问题" | F0-A 五诊断: 单位失配(ψ 惰性, distort 0.1%)+ easy-target bias(弱目标 −1.0 vs 易目标 2.5e7)+ 归一化修复 | Case 2+3, 4.94 @ (12,6) |

## 27. 待办(理论侧)

1. 将阶段 20 的计数条件上界、Rao-Blackwell 分层、`(1-1/e)` 贪心界补入 `FORMAL_PROOFS.md`(当前仅 README 记录)。
2. 为 NOMP refine 的 Top-L 剪枝补充 worst-case 论证或降级为"经验剪枝"。
3. 统一 `_uncoverable_targets` 容差与贪心接受准则的 `eps` 语义。
4. 将阶段 21 的 LLR 漂移恒等式、擦除线性缩放、DP 收缩链、`Chernoff <= min(I+, I-)` 与 FFT 卷积顺序 `P_D` 精度论证补入 `FORMAL_PROOFS.md`。
5. 阶段 22 已部分补入(定理 4.91 floor-cover、事实 4.92 NP 可容许性、事实 4.93 I+(b) 单调);待补: `var1>var0 ⇒ {LLR>t} 双侧窗口` 结构与贪心注水 4.7% 界的正式论证(后者现记录于 non-claims)。
6. **分配-时间闭环(已完成,负结果)**: floor-cover 精确分配接入主动检测循环已实施于阶段 23 —— 预算 Bellman G 值调度最差目标 T 3.50 vs active(τ_pred)2.08(增益 -40.7%): 价值函数的代价目标与 P_D 阈值目标不一致,不构成对闭环思想的否定性定理,只构成对该实例化的否定;目标校正后的正结果见阶段 24。
7. **阶段 23/24 定理补入 `FORMAL_PROOFS.md`**: 预算状态值递归与预算单调、Blackwell 一级/二级剪枝的精确消去、残差 z 标准化与 τ 收敛性、双阈值数值校准、ν 加权 min-max 与约束内嵌、deployable 家族 gap 论证(现记录于 `THEORY_DEVELOPMENT.md` §1.15/1.16 与 `SYSTEM_MODEL.md` §12)。
8. **F0 定理补入 `FORMAL_PROOFS.md`(阶段 25)**: 信息结构隔离的差距分解定义、通信域信念的量化核校准有效性、token bits 可行性的单调性边界(4-bit 不可行为实例实证而非定理)、拥塞价格 ψ 的稳定性语义。
9. **F1/F2 冻结,由规模审计结果决定(advice/006)**: 不再预先承诺 token 消融/协调算法对比/稀疏 U2U/动态 owner。规模审计(阶段 26)已给出第一个瓶颈:**检测层延迟增长(最大档)→ 下一步唯一问题是 target allocation / resource competition**;若其后 rx 负载(19×(K−1) bits/周期,线性)先成为约束,再做稀疏/结构化 U2U。待对应方向启动后再细化 F1/F2 内容。
10. **规模审计扩展(如需要)**: 多场景抽样(每档 3+ 场景种子,消除弱目标占比随 Q 下降的组成效应)、K=24/32 继续外推、rx 负载显式建模(接收侧解码/处理成本)—— 仅在审计结论需要加固时启动。