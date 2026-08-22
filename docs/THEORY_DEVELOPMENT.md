# 理论发展文档(理论框架总览)

- 日期: 2026-08-17
- 定位: 按**主题**组织当前完整的理论框架(静态视图);循序发展过程见 `docs/THEORY_DEVELOPMENT_HISTORY.md`
- 证明细节: `docs/FORMAL_PROOFS.md`(全部定理/引理的完整证明与实现-测试追踪表)
- 系统模型与记号: `docs/SYSTEM_MODEL.md`

---

## 1. 理论体系总览(主题视图)

### 1.1 融合理论(Gate G3)

- **偏转最优线性得分**: 收到集合 `R` 的权重 `w = Sigma0_R^{-1} delta_R`(引理 2.1 白化、引理 2.2 紧化归一)。
- **KKT 表示**: 在 `P_D > 0.5` 工作点,全局线性得分最优解的权重位于单参数族 `{w(mu) = L^{-T}(Q + mu I)^{-1} L^{-1} delta : mu >= 0}`(定理 2.3);因此一维搜索即可覆盖全局最优。
- **集合单调性**: 最优 `P_D` 对收到的报告集合单调(定理 2.4,零延拓可行域嵌套,与 KKT 解耦)。
- **比例协方差闭式**: `Sigma1 = c Sigma0` 时 `P_D = Phi((sqrt(D) - z_FA)/sqrt(c))`(定理 2.5)。

### 1.2 期望-P_D 与次模性(Gate G4)

- 期望保单调(定理 3.1): 固定图样的 `P_D` 单调,期望是非负混合。
- 可分解偏转与凹性(引理 3.2/3.3): 对角 `Sigma0` 下 `D(S)` 模化,闭式 `P_D` 在 `c + D - z_FA sqrt(D) >= 0` 区域凹。
- **有界区域次模性**(定理 3.4): 期望保持次模 → 基数贪心保留 `1 - 1/e` 界。

### 1.3 精确选择理论(Gate G5/G8)

- 精确等成本配额选择(定理 4.7)、**异构成本预算选择**(定理 4.7A)、**精确 max-min 选择**(定理 4.7B)、Pareto 支配剪枝(引理 4.7C)。
- 最小成本门限 B&B(定理 4.7D)与 Cauchy 上界(引理 4.7G)、缩放 max-min 可行性证书(定理 4.7E,`O(Q log O log V)` 阈值可行性,引理 4.65)。
- 精确性假设: 目标可分、加性报告成本、无跨目标耦合;组合精确性与离散化值预言机/一维融合搜索分离(对应审稿 P0-7 的"Exact"措辞修正)。

### 1.4 RIS 信道、量化与部署(Gate G5/G9-G13)

- 加性功率增益永不有害(§4.1)、相位量化损失界(§4.2)。
- 网格搜索次优性界(定理 4.3)与 **Lipschitz 分支定界 epsilon 最优证书**(定理 4.4,引理 4.5 坐标 Lipschitz)。
- 子阵孔径守恒(引理 4.8)、坐标上升(引理 4.9)、孔径-开销权衡(引理 4.10)、闭式孔径最优(定理 4.11)。
- **max-min 偏转注水**(定理 4.12): 显式、多项式、弱目标获得更多孔径;精确块代理(引理 4.13)。

### 1.5 系统级局部最优证书(G14-G18)

- 贪心感知系统级局部最优(定理 4.14)、单元素证书(定理 4.15)、有界多块证书(定理 4.16)。
- **联合布放-分配局部最优**(定理 4.17);有限终止与显式复杂度(定理 4.24);`Q > 3` 缩放(定理 4.25)。均为局部证书,非全局最优。

### 1.6 分布式共识与计数理论(G19-G24, G40-G43)

- 比特粒度可行性(引理 4.18)、分布式门限优化(引理 4.19)、对等多数(引理 4.20)、多跳可达(引理 4.21)、公共故障与异构可观测性(引理 4.22)、多数扩展(引理 4.23)。
- 稀缺报告比特下的共识优势(引理 4.40)、多数奇偶边界(定理 4.41)、优化本地门限(推论 4.42)。
- **精确泊松-二项可行性**(定理 4.43)与精确最小多数计数(定理 4.43A): 对等多数 P_FA/P_D 是精确泊松-二项尾,可行性判据自 `M=6` 起严格。

### 1.7 信息预算与架构切换(G44-G50)

- 软融合内信息预算单调(定理 4.44);朴素闭式律失效(引理 4.45);**精确有效信息坐标 `rho_exact`**(引理 4.46,原始 rho 高估 2.38-2.78x)。
- 双分支架构切换(引理 4.47)、逐目标切换支配全局切换(引理 4.48)、软报告比特再分配单调上升(引理 4.49)、受限目标模式上升接受律(引理 4.50)、逐帧模式上升推向最差时序 QoS(引理 4.51)。
- 迟滞重构损失界(引理 4.56)与成本感知迟滞选择(推论 4.57)。

### 1.8 移动性与预测(G26, G51-G54)

- 时变几何下的自适应注水(引理 4.26);AR(1) 条件均值 RIS 预测(引理 4.52)、h 步 MMSE 预测与误差协方差 `1 - rho^{2h}`(引理 4.53)。
- **负结果**: 期望增益梯度上升单调(引理 4.54)但代理非系统最优(推论 4.55)——量化下精确最差 `P_D` 从 0.7200 降到 0.6557,保留 MMSE 相位。

### 1.9 鲁棒分配理论(4.58-4.78)

- 精确最坏情形机会约束分配(定理 4.58);BSC 退化排序与精确 LRT ROC 支配(定理 4.59);擦除随机单调与期望-P_D 单调(定理 4.60);速度有界移动包络(定理 4.61/4.61A)。
- 独立逐目标模糊归约为标量 DP(定理 4.62)、精确鲁棒 DP 复杂度(定理 4.63)、物理报告链路模型(引理 4.64)。
- 联合功率-比特选项集含单维基线(引理 4.66)、向量化枚举(引理 4.67)、感知/通信信道解耦(引理 4.68)、通信感知得分证书最优代理(引理 4.69)。
- **通信模糊端点归约**(引理 4.70): 矩形端点 `(flip_hi, success_lo)` 即最坏情形 → 端点归约鲁棒 DP(推论 4.70A)与鲁棒联合分配精确性(引理 4.71)。
- WTA 功率分配(引理 4.75/4.76)、误差反馈纠正胜者选择(引理 4.77)、UCB 证书停止(引理 4.78)。

### 1.10 NOMP 式在线优化理论(4.79-4.82)

- 每目标最小覆盖(引理 4.79);**leximin refine 单调与终止**(引理 4.80);单交换注水可达性(推论 4.80A);信道失配下的通信感知 refine(引理 4.81);QoS 缩放 leximin refine(引理 4.82)。
- 性质: 最差目标值单调不减,有限步终止于局部最优或硬轮数上限;无全局最优性证明(数值上与小规模 exact-frontier 一致)。

### 1.11 学习-优化耦合(4.83-4.90)

- 两阶段 MAPPO-NOMP 分解(引理 4.83)、MAPPO-NOMP 适配器(引理 4.84)、UCB 模式选择(引理 4.85)、优先级中间件(引理 4.86)、NOMP 终值奖励塑形(引理 4.87)、多温度提案集成(引理 4.88)、难度自适应课程(引理 4.89)、UCB 温度分配(引理 4.90)。

### 1.12 最新: 擦除期望上界与 Rao-Blackwell 估计(2026-08,README 记录)

- **Rao-Blackwell 分层 MC**: 接收计数律精确(泊松-二项),按计数分层 `E[P_D] = sum_n P(N=n) E[P_D | N=n]` 无偏,方差 `E[Var(P_D|N)]/samples` 不超过普通 MC(全方差律,实测低约 40%)。
- **计数条件上界** `count_conditional_upper_bound`: `E[P_D] <= sum_n P(N=n) max_{|S|=n} P_D(S)`,几乎必然且逐点不松于无擦除上界;最大 n 子集由 n 个最强报告取得(偏转单调 + 独立对角协方差);随机实例平均收紧约 20%。
- **贪心界**: 单目标基数受限激活达到 `(1 - 1/e)` 最优(经典次模性论证);`verify_submodularity` 等数值校验。
- 注: 以上结果目前记录于 README,`FORMAL_PROOFS.md` 的对应定理待补。

### 1.13 最新: 检测信息与主动检测主线(2026-08,advice/001.md 驱动,Gate A/B)

- **LLR 漂移恒等式**: 链路检测信息 `I+ = E_H1[LLR] = KL(p1||p0)`,`I- = -E_H0[LLR] = KL(p0||p1)`,构造性精确(Wald 序贯观点下即每观测证据增长速率)。
- **DP 收缩链**: `I_post <= I_quant <= I_sensing`(量化=BSC=擦除均为信道);尾质量采用无取消计算(右侧 `Phi(-a)-Phi(-b)`)+ 1e-300 下限,400 随机实例数值全部成立。
- **擦除线性缩放**: 可检测擦除符号对 LLR 贡献恰为 0,故 `I+ = s * KL(p1_rec||p0_rec)` 精确(非近似)。
- **Chernoff 信息**: `C = max_{s in [0,1]} -log sum_y p1(y)^s p0(y)^(1-s) <= min(I+, I-)`;Gate A 实测对未来 `P_D(4)/P_D(8)/n*`(P_D*=0.9)的 Spearman 预测力 0.987/0.998/0.994,KL 0.978/0.976/0.975,全面超越偏转代理(0.69-0.74,印证 G1-A 的 0.588)与单步 `ΔP_D(1)`(0.85)。
- **精确顺序检测**: LLR PMF 按 FFT 卷积 n 重(统一网格 + 每步重定心),`P_D(n)`/`P_FA` 无高斯近似;异构观测序列(不同报告/功率)按序卷积;NP 阈值 = H0 上尾 ≤ α 的最小网格点;与暴力枚举逐点一致(1-ulp 双子原子合并)。
- **主动检测策略**: 每周期将动作分配给 `tau_pred = (eta(n+1) - n*I+)/I+` 最大的目标,目标内取 `I+/cost` 最佳报告-功率;Gate B(预算紧张,2 目标 × 3 报告 × 2 功率)最差目标平均 T **2.08** vs static 3.10 / myopic 2.33,终态最差 `P_D` **0.599** vs 0.234/0.564;预算松弛时三策略等价(如实记录)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.4,`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.4)。

### 1.14 最新: 检测感知量化与信息梯度(2026-08,advice/001.md 驱动,Gate C)

- **信息-比特单调性(可证)**: `I+(b)` 随比特数单调不减 —— 均匀细化量化器族下 `rec(b)` 是 `rec(b+1)` 丢 LSB 的确定性函数,DP 链给出 `I+(b) <= I+(b+1)`;flip/success 单调性经 4.59 级联恒等式 + DP 验证。`I+(0) = 0`(单 bin 无判别力)。
- **诚实否定: 凹性不成立**。`I+(b)` 边际在 b=3 处跳升(0.163→0.465,步长细化越过 H0 主体时判别力集中释放),故**不宣称注水精确最优**;贪心 vs 暴力枚举相对差距 ≤ 4.7%(诚实启发式界)。
- **设计度量层级(Gate C 核心教训)**: 跨设计比较时 `I+`(均值漂移)会误导 —— span 扫描中 62.5% 实例 `argmax I+` 的 span 使 `P_D(4)` 低于最优 1% 以上(粗量化集中 H0 质量,经 BSC 后 KL 反而高,但精确 `P_D` 不跟随);**Chernoff 正确跟踪 `P_D`**(仅 14.6% 失败)。层级: 精确 `P_D` > Chernoff > `I+`。
- **信息梯度分配的否定结果**: `sum-I+`/`maxmin-I+` 注水分配的最差目标 `P_D(4)`(0.463/0.507)远低于精确 max-min 分配(0.794,100% 实例占优)—— 跨设计(比特数即设计)排名失效使 KL 梯度分配不成立;**Chernoff 最优分配 0.772 且在 67% 实例与精确分配一致**,精确枚举为基准。
- **1-bit LLR 结构**: `var1 > var0` 时 LLR 凸,`{LLR > t}` 为双侧窗口 `{x<a} U {x>b}`;`var1 < var0` 时为单区间;等方差退化。左臂 H1 质量可忽略(μ1>μ0),窗口对单阈值 KL 增益均值 0.00017(物理区间如实报告,设计上保留单阈值)。
- **精确 max-min 分配(floor-cover 定理)**: 最差目标 `P_D(n)` 的 max-min 分配 = 最小成本覆盖 —— 候选水平 `L` 下目标 `t` 的最小成本 `c_t(L) = min_o min{b: f_{t,o}(b) ≥ L}`,最优值 = 满足 `Σ_t c_t(L) ≤ B` 的最大 `L`;**无凹性/单调性假设**,多项式复杂度;Gate C 实测与小规模穷举 100% 一致(0.794),而 I+ 注水(0.463/0.507)从不一致。
- **NP 可容许性(单调性定理及其边界)**: 细化链下 `b` 比特观测是 `b+1` 比特的确定性函数(丢 LSB),故 `b` 比特的任意 α 水平检验是 `b+1` 比特上的合法检验,更细字母表的 LLR 检验在 α 水平下最有力 ⇒ 真实 `P_D(n)` 随比特数单调不减(flip/success 同理,经 4.59 级联);**但网格统计量(`rint` 原子与累加)非精确 LLR**,实测违反 ≤ 0.008(更细检验取更保守 P_FA 时),`verify_pd_bits_monotonicity` 如实报告;floor-cover 不依赖单调性,仍精确。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.5,`FORMAL_PROOFS.md` 定理 4.91/事实 4.92 已补(见 HISTORY §27.5)。

### 1.15 最新: Belief-Bellman 主动控制(2026-08,advice/002-003.md 驱动,Gate D1)

- **显式预算状态 Bellman** `V_t(pi, B)`(`active_detection_bellman.budget_bellman_value`): 状态 = 后验 log-odds `l` × 剩余观测预算 `b`(整数成本单位),递归
  `V_h(l,b) = min( stop0, stop1, min_{a: c(a)<=b} [ c(a) + E_Y V_{h-1}(l + llr_a(Y), b - c(a)) ] )`;
  续集为空(预算付不起任何动作)时强制停止,停止本身不耗预算;`V_0` 为终端停止代价。宽松预算(付得起任意动作)时与无约束网格值逐点一致(≤1e-9),预算单调 `V_h(l, b+1) <= V_h(l, b)`。
- **Blackwell 三级剪枝的前两级**: 第一级 exact dominance(LP 可行性 `p_h^b = p_h^a K`,`c(a) <= c(b)` 时 `b` 永非 Bellman 最优);第二级 `value_bound_prune` 精确动作消去 —— 动作 `a` 的续估 `c(a) + E_a V(l + llr)` 在整条网格上从未优于终端停止代价 `min(c01 pi, c10 (1-pi))` 时当步剪除(保守化使用同预算层 `V_{h-1}(l,b)`,因 `V_{h-1}(l,b-c) >= V_{h-1}(l,b)`,消去保持精确);与无剪枝值函数逐位一致(测试),零信息高代价内核全步剪除。第三级 approximate search 未做。
- **残差自适应(Reflexion 数学化,advice §4)**: 每次观测结算 Bellman 残差 `r_t = c(a_t) + V(l_{t+1}) - V(l_t)`;残差非假设无关(价值是信念混合、实现抽自单假设),故对 H0/H1 两个模型条件分布标准化 `z_H = (r - mu_H)/sigma_H`,`mu_H = c + E_H[V(l+llr)] - V(l)`,`sigma_H^2 = Var_H[V(l+llr)]`;累积均值(收敛估计,非 EMA)统计量 `tau = min(|mean_0|, |mean_1|)` —— 单假设流下正确模型的活动假设 z 均值为零,`tau -> 0`;模型不可靠时两个条件标准化同时偏移,`tau` 增长。`tau > margin` 触发 robust(一步前瞻)模式,持续 `explore_rounds` 步后转 explore(最大 `I+/cost`),`tau < margin/2` 迟滞回归。
- **Gate D1b 判定(诚实)**: Q=1、R=3、b=1..4、P=2 档、H=B=6、c10=c01=20 下,精确预算 Bellman 相对最优 myopic(dpD,总代价 5.062)仅 +0.4%,预算扫描 B=2..8 全部 ≤2.7%(B 越大增益越大,0→2.7%),未达 5% 实质性门槛 → **单目标尺度 "Agent 化" 不值得做**;Bellman/dpD 相对信息型 myopic(τ_pred/Chernoff/c,6.2-6.5)稳定 15-20% —— 瓶颈在信息型调度层而非多步前瞻。
- **Gate D1c(失配检测,诚实)**: 正确模型 τ=0.038 不触发;success 0.95→0.5 擦除失配 τ=0.666 触发并切模式;flip-only 失配不触发(统计量局限: 翻转不改 LLR 原子幅度,如实报告);适配后实现代价 ≤ naive Bellman(MC 容忍内)。
- **分配-时间闭环(负结果,诚实)**: Gate B 逐周期预算循环中,预算 Bellman G 值调度 `G = V(l,b) - [c + E V(l+llr,b-c)]` 最差目标 T 3.50 vs active(τ_pred)2.08(增益 -40.7%);价值函数的停止经济学(代价目标)与 P_D 阈值目标不一致,价值代理(I+ 漂移)高估进度 —— 确认 D1b 判定在逐周期多目标层面同样成立。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.6;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 23;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.7)。

### 1.16 最新: 目标对齐的序贯检测(2026-08,advice/004.md 驱动,Gate D2)

- **诊断(P0,目标函数校正)**: Gate D1 的成本 Bellman 最小化"采样成本 + 贝叶斯误差",但系统目标是约束检测延迟 `min_Pi max_q E_1[T_q]` s.t. `P_FA <= alpha`, `P_MD <= beta`。目标错配正是旧闭环负结果(3.50 vs 2.08)的原因。校正后的单目标延迟 Bellman(`delay_value_iteration`):续观分支每周期恰付 1(检测周期),声明分支由对偶价格 `(xi, zeta)` 定价,`V_h(l,b) = min{ zeta*pi, xi*(1-pi), cycle_cost + min_a[lam*c(a) + E V_{h-1}(l+llr, b-c(a))] }`。
- **P1(真双阈值停止)**: `calibrate_sprt_boundaries` 数值校准 `(A*, B*)`(围绕 Wald 值网格扫描 + MC,满足约束中 `E_1[T]` 最小;离散量化+BSC+擦除核下 Wald 近似不精确:校准 A*=2.34/B*=-3.54 等实测优于 Wald,且 Wald 边界在预算紧时违反约束)。`T_q` 从此是严格 stopping time,替代 P_D(n) checkpoint。
- **Gate D2-A 判定(单目标,Q=1, R=3, b=1..4, P=2, H=10, B=16, α=β=0.05)**: 目标对齐后 delay-Bellman `E_1[T]=3.85` 相对 one-step exact-P_D 动作选择 `2.43` 慢 58%,相对信息型 myopic(4.59)快 16% —— **单目标无 long-horizon 规划价值**(advice Case 1 方向);旧成本 Bellman 2.05 但 P_FA=0.102/P_MD=0.118 违反约束(如实,目标错配的代价)。
- **Gate D2-B 判定(Q=2,strong 10/16-flip0.02-succ0.98 + weak 7/11-flip0.08-succ0.9,共享预算 30,min-max via ν)**: 精确联合序贯 oracle(`joint_delay_value`,状态 `(l1,l2,B)`,每周期一个动作,停止边界内嵌价值函数 —— 达标即停、带内停止不可行,消除价格-边界错配)经 ν 权重扫描 `max_q E[T_q] = max_nu sum nu_q E[T_q]`:**最差目标 H1 延迟 5.57 vs myopic ΔP_D 6.48(+14.1%)vs static floor-cover 11.39 vs τ_pred 16.98,超过 5% 实质性门槛 → Case 2 正结果:多目标竞争创造规划价值**;joint 的误差 (0.033, 0.077) 与 myopic (0.045, 0.085) 相当或略优(weak 目标 P_MD 略超 0.05 为共享预算资源限制,所有策略同样,如实报告)。
- **边界内嵌的关键**: 早期版本(价格停止 + 边界覆盖)joint 8.31 反超 myopic —— 价值函数的停止语义与 rollout 边界错配(strong 达标后仍被联合价值"等待");约束内嵌(terminal = 0 于边界外、1e9 于带内)后 joint 5.57 且 strong P_MD=0.033。教训:约束必须进入动态规划,不能后置覆盖。
- **D2-D 可部署控制器(advice §6/Case 3,部署而非 oracle)**: 精确联合 oracle 状态空间指数级,不可部署;`make_deployable_controllers` 提供与 Q 线性复杂度的家族 —— dual G 值调度 `argmax_{q,a}[nu_q(V_q - (c + E V_q)) - lam c]`、Whittle 指数 `argmax (V_q - (c + E V_q))/c`、一/两步 rollout(同伴等待或下周期最优观测),共用每目标延迟值(构建 O(Q),一次)与校准边界。**预算充足(45)时:Q=2 dual 最差目标延迟 4.39 vs oracle 4.23(部署 gap 仅 +3.9%,捕获 96% 规划价值)vs myopic 4.93(+10.8%);Q=3 rollout_1step 6.81 vs myopic 7.43(+8.4%,规划价值扩展到三目标)**;每周期决策 2.5-6.5ms、单目标值构建 2-4s,实时可部署。资源不足区(B=30,三目标)差距收敛(myopic 持平),如实报告。`results/d2_deployment_gate.json`。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.7;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 24;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.7)。

### 1.17 最新: 分布式信息审计与系统纠偏(2026-08-17,advice/005.md 驱动,Gate F0)

- **系统定位纠偏**: 项目正式定位为"通信受限条件下的分布式多 UAV 任务驱动协同检测";停止 Exact-first / Fusion-center-first / Algorithm-first 三条惯性,改为 Distributed + Deployable + Task-Oriented。集中融合中心只作离线审计 oracle;**部署动作约束 `a_{i,t} = π_i(I_{i,t})`(信息集 = 本地观测 + 成功送达 token + 自身历史)成为 P-DIST 问题的关键边界**;资源从单一全局预算改为节点级 `(B_i^U2U, E_i^sense, T_i^comp)`;U2U 协调轮次 ≤ 2、不要求共识收敛;证据通信 vs 协调通信拆分。
- **F0 审计结构(信息结构隔离)**: 决策规则固定(局部 dual G 值 `J_{i,q,a} = ν_q[V(l,b) − (cycle + c + E V(l+llr,b−c))] − λc` + 邻居 intent 拥塞价格 `ψ = −η·count(intents)`),仅变化信息结构: A centralized(全局 belief,oracle)/ B full_message(精确证据 token)/ C compact_token(19-bit 量化 token,LLR 5 bits)/ D local_only(零通信)。每目标固定 owner,owner belief 按逐模式校准双阈值停止。
- **逐模式校准(方法学,印证 D2 教训)**: C 的 belief 在**通信域**维护(自身与接收证据同过 token 编码器),双阈值在量化核上校准;若沿用精确核阈值,中值量化会**夸大漂移**(4-bit 时精确核阈值下 P_MD 0.065 违反约束)。**校准稳定性是审计协议的一部分**: 单级 MC 校准在近并列可行边界间翻转,曾使 Δ_comm 符号为负(伪影);两阶段校准(粗扫 scan 300 + 高 MC 局部复选 verify 2000,固定种子)消除该伪影,残余校准种子敏感性如实报告。4-bit token 在 α=β=0.05 下校准不可行(证据塌缩),5/6-bit 可行 —— infeasible region 实证(advice §20 Contribution 4)。
- **F0 判定(K=6,Q=3,4 seeds × 800 runs,两阶段稳定校准)**: Δ_decentral(B−A) = +0.77 周期(+17.1%)→ 去中心化代价实质存在,研究重点为 ownership/冲突消解/分布式价格;Δ_comm(C−B) = +0.11(+2.0%,校准种子敏感性范围 +1.8%…+6.3%)→ **C ≥ B 方向与信息论一致,量化损失小但非零,略低于 2% 实质门槛**(如实);Δ_coop(D−C) = +27.2(+507%)且 D 无法满足错误约束(P_MD 0.48)→ **协同是必要的**,不只是有益。
- **计算性能**: 局部 dual G 值单 UAV 单周期决策 ~0.42 ms(无 rollout,优于 D2-D 的 2.5–6.5 ms);单目标延迟值构建 ~0.6 s(线性);K=8/Q=4 扩展保持 A < B ≈ C << D 结构且错误约束满足 —— 分布式调度器实时可部署。
- **分布式稳定性指标(advice §18)**: conflict 率、duplicate sensing 率、role switch 率(固定 owner 为 0)、belief 分歧 `D_L(t)`(B/C ~0.75–0.78,D 3.44);belief 允许不一致,要求的是动作不长期振荡冲突。
- **模块重分类(advice §10)**: Exact/联合 oracle → 离线审计;Bellman/dual/rollout → 局部动作价值层(保留为分布式调度器的 J 索引);NOMP → "有界局部修复"候选(F2 公平对比后决定去留);MAPPO → 仅允许 `a_i = π_θ(I_{i,t})` 的 CTDE 部署 actor(F2 候选)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.8;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 25;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.7)。

### 1.18 最新: 规模审计(2026-08-17,advice/006 驱动,Gate F0-S)

- **收敛声明**: 论文主线锁死为"通信受限分布式协同检测",唯一主问题为"紧凑证据交换和局部任务决策能否在 UAV/目标规模增加时维持可靠检测";FOV/安全距离/动态 owner/mobility/coalition/稀疏拓扑全部冻结为假设与后续扩展。全部机制冻结(固定 owner、full mesh、19-bit token、dual-G + 拥塞价格、当前序贯检测与场景生成),唯一变化 (K,Q) ∈ {(6,3),(8,4),(12,6),(16,8)},K/Q=2。
- **五个指标**: J = max_q E₁[T_q](主线 compact-token)、P_MD^max、P_FA^max、B_U2U/UAV(发送/接收)、T_decision/UAV;稳定性指标降为诊断。
- **Gate A(检测稳定)未通过**: J(C) 5.37 → 6.08(+13.1% > 10%),P_MD^max 0.0704((12,6) 处 0.070 略超 β+2pp),P_FA=0;J 非单调((8,4) 3.97 最优),增长集中于分布式层(集中 oracle 仅 +5.6%,B +12.7%、C +13.1%)→ **第一个瓶颈:检测层延迟增长,下一步研究 target allocation / resource competition**。
- **Gate B(局部计算)通过**: T_decision/UAV 0.50→0.89 ms,亚线性于 Q 且与 K 无关(每 UAV 只评估自身核的 Q 目标),本地 dual-G 计算不是瓶颈。
- **Gate C(通信扩展)通过(结构性发现)**: tx/UAV 恒 19 bits(单广播);**rx/UAV = 19×(K−1) 线性增长(95→285 bits/周期)** —— 全 mesh 接收侧负载是通信扩展的结构性成本,若其先于检测层成为约束则下一步为稀疏 U2U。
- **协同必要性随规模强化**: 零通信基线 (16,8) 下 P_MD → 1.0(40 周期零声明),D 的 J → 40(饱和)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.9;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 26;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.19 最新: 目标竞争审计与分配修复(2026-08-17,advice/007 驱动,Gate F0-A)

- **诊断(不改机制,5 个逐周期量,paired 规模)**: r_min −44%、H_idle^max +31%、n_max=K(全集中)、ρ_alloc −62%(0.442→0.168)、distort≈0.1% → **Case 2+3(starvation + over-concentration)双签名**。
- **单位失配定理(实证+理论)**: 带内 dual-G 增益 ~1e5–1e9(停止带 1e9 终端),O(1) 加法价格/age **数学上不可比**(distort 0.1% = ψ 几乎不改变选择)→ **加法协调项有效 ⟺ 决策变量有界**;归一化(每 UAV 每周期按本地增益尺度)后价格/age 恢复有效。
- **Easy-target bias(机制)**: 带内增益 ≈ 一步穿越概率×1e9 —— 弱目标增益 −1.0 vs 易穿越目标 +2.5e7,索引按"立即可解决性"排序,系统性低估难目标,min-max 语义丢失(ρ_alloc 下降根因)。
- **修复(一次一个标量)**: 原样 age/γ 无效(单位失配);归一化索引 +(12,6) η=1 价格 → **J 5.56→4.94(−11%,P_MD 达标)**;(16,8) 全配置无效 —— 最大档 worst-target 由**单个内在困难目标**决定(16/8 的 q7 e1=5.68 独大;12/6 的弱目标 q0=5.61),按 advice §9 属目标难度分布主导,非分配可修复。
- **理论**: 并发观测边际价值 ∝ n⁻²(延迟 ~ A/(n·I⁺))⇒ 凸价格(γ≥2)正确但需先解决单位;age 项 = 任务级 AoI(检测 AoI,顺序检测怕的是 belief 信息流长空隙);修复全程在 19-bit token 协议内,感知能量机会成本与检测信息下界 `E_1[ΣI⁺] ≥ d(1−β‖α)` 均不受影响。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.10;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 27;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.20 最新: 目标难度分解(2026-08-17,advice/008 驱动,Gate F0-D)

- **三层分解(同一 realization,纯诊断)**: `J_q^dist = J_q^iso + Δ_q^comp + Δ_q^dec` —— J^iso 只留目标 q(同 UAV/核/owner/资源/阈值)、J^cent 集中 oracle、J^dist 部署系统;(6,3)/(12,6) 最难目标 **Case B 竞争主导**(Δ^comp 60%/59.7% 占比),(16,8) 弱目标 **Case C 分布式主导**(iso 29.3% / comp 32.4% / dec 38.3%)。
- **无极端样本效应**: {J^iso} 中位数 1.99→1.22→1.13、最大 2.10→1.80→1.75 随 Q 不升反降(弱目标固定、占比缩小的组成效应)→ F0-S 的 J 增长非 max-over-Q 伪影。
- **dec gap 成分(决定性)**: (16,8) 弱目标量化损失 22.3% > 投递/局部 16.0% —— 分布式 gap 是**证据保真度主导**(弱目标小 LLR 原子被 5-bit 中值量化压缩,与 4-bit 不可行同源),非分配问题(解释 F0-A 分配项修不动 16/8)。
- **信息论下界 sanity check**: T_LB = d(1−β‖α)/I+_max = 3.79 周期/观测(16/8 弱目标),周期下界 ≈ 0.24 ≪ J^iso 1.75 → **远未到信息物理极限**;难度指纹 I+_max 0.700、Chernoff 0.130、N_useful 16/16(非有效 UAV 不足)。
- **理论**: 分解为数值差值审计(非因果独立项,paired 解释力);信息下界 `E_1[ΣI⁺] ≥ d(1−β‖α)` 给出可优化性判据:J^iso ≫ T_LB ⇒ 还有证据路径 headroom;难度感知权重被明确反对(接近物理不可达的目标会成为 resource sink,须先由 J^iso 区分 hard-but-improvable 与 intrinsically-uninformative)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.11;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 28;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.21 最新: Token 保真度审计与分解方法学修正(2026-08-17,advice/008 后续,Gate F0-E)

- **Token 保真度负结果(干净)**: 同一 19-bit 预算内测试三个有理论支撑的设计 —— Lloyd-Max(质心条件=每 bin 无偏、H1 加权)、μ-law 压扩(近零细分)、范围匹配 uniform、以及 **L̂ 5→10 bits 预算内重分配**(F0-A 证明 u/r/χ 死载荷)。配对固定阈值下全部 ≤±0.1 周期;**L̂=10 bits(近精确)在 16/8 仍无增益(−0.2%)** ⇒ 量化速率不是瓶颈。
- **F0-D 方法学修正**: 原"量化份额 22.3%"是 B 模式未归一化索引的混淆;修正(全参考用归一化索引,cent 无价格 η=0、B 保持 η=1)后 (16,8) 最难目标: iso 29.3% / comp 34.5% / dec 36.2%(Case C 不变);dec 成分 **投递/局部 +40.6% > 量化 −4.4%**(中值量化漂移放大伪影方向)。
- **理论**: 中值量化每 bin 有偏性(漂移放大);率失真 D(R) 在 R=10 已近零而 J 不变 ⇒ 损失在传输路径(投递失败 + 意图 1 周期滞后),不在编码;F0-A 死载荷诊断支持预算内重分配但收益 ≈0。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.12;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 29;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.22 最新: 投递/协调审计与尺度自适应价格(2026-08-17,advice 后续,Gate F0-F)

- **dec gap 分解((16,8))**: 完美投递 −1.2 周期(投递=主导项,物理层受限);**新鲜意图(R_coord=2)负结果**(+2.0~+2.5,价格过度驱散)。
- **尺度自适应价格(正结果)**: 归一化主线下 (12,6) η*=1(J 4.935)、(16,8) η*=0(J **6.014→5.680,−5.6%**,P_MD 0.054 达标) —— 拥塞价格最优尺度随 K 塌缩;大 K 下集中=高效聚焦(弱目标延迟 ∝ 1/n),价格只应惩罚超出 K/Q=2 的浪费性重复。
- **理论**: 并发观测边际价值 ∝ n⁻² ⇒ 大 K 聚焦是 min-max 最优,固定尺度价格在 K 增大时把高效聚焦一并驱散;标量索引参数,协议/预算/模型零改动。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.13;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 30;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.23 最新: 操作点校准 / 多场景统计 / FRIDS 原始-对偶调度(2026-08-17,advice/009 驱动,Gate F0-G1/G2/G3)

- **F0-G1(策略匹配操作点)**: 部署策略下坐标扫描 A → (12,6) J 4.94→3.37(−32%)、(16,8) 5.68→4.58(−19%) —— 阈值保守是最大单一可修复损失;P_FA 结构性 ≈0(H0 流强负漂移,约束由 P_MD 绑定)。
- **F0-G2(多场景统计)**: 5 场景/档,场景方差占 69-95%;16/8-vs-12/6 平均增长仅 **+3.9%**(单场景 +13% 为偏差抽样)。
- **F0-G3(FRIDS)**: 从 min-max 检测缺陷松弛 `max z s.t. Σx·g ≥ zD, Σx ≤ 1` 导出本地索引 `J = ν_q·g_{iq}`,g = 后通信 I+ × 投递可靠性(感知×通信进同一边际价值);ν 投影次梯度 `ν ← Π_Δ[ν+μ(D̄−S̄)]`(自生价格,**η(K) 退役**);信息可行性证书防 resource sink;策略匹配降 B 修复 H1 下 H0 误判。**生死门通过**: (12,6) 5.16→2.75(−46.7%)、(16,8) 5.36→3.26(−39.3%),P_MD ≤0.066、5/5 场景全胜、无回退。
- **理论**: 对偶价格 ⟺ 检测缺陷(Detection-deficit duality);可靠信息调度(Reliable-information scheduling);可行性感知公平(Feasibility-aware fairness);规模无关协调(Scale-free coordination)—— 四个创新点取代"归一化+尺度价格"启发式。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.14;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 31;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.24 最新: FRIDS 一致性审计与 FRIDS-v2(2026-08-17,advice/010 驱动,Gate F0-G4)

- **审计闭合(五查)**: ① `g = I+^post·s_u2u ≡ KL(最终可观测核)`(数值精确,核内擦除与 U2U 投递独立信道,无重复计数,恒等式 `I+^final = s_u2u·I+^post` 固化测试);② ν 单纯形与 zD_q 约束对偶不一致 → **需求归一化重参数化**(`Σx·g/D_q ≥ z`,对偶回到普通单纯形,本地索引 `J = y_q·g_{iq}/D_q`);③ D−S 量纲 → **归一化服务比** `S/(D+ε)`;④ provenance: v1 读 owner belief(全局)→ v2 **严格局部**(每 UAV 自己的 D/S/y,全部 ∈ I_{i,t});⑤ token 记账: `b_q = ⌈log2Q⌉`(Q=8→3 bits),死载荷 u/r 腾出,总长 17 ≤ 19。
- **FRIDS-v2(采纳)**: 指数梯度(mirror descent)`y ← y·exp[μ(r̄−S/(D+ε))]/Σ`(天然 y>0,nu_floor 退役);10 场景 @(12,6)/(16,8)、5 场景 @(6,3)/(8,4),策略匹配 B:**(16,8) 4.16→3.53(−15.1%)、(12,6) −7.6% 无回退、全档 P_MD ≤0.053 零违例、胜率 0.9** —— 严格局部+对偶一致版本既更理论干净又更快。
- **信息负载 cut(Theorem 4)**: `ρ(S) = ΣD^info/(H·Σ_i max_{q∈S} g_{iq})`,ρ>1 ⇒ 子集不可按时完成;实测 ρ_full ≈0.03-0.05 ≪ 1(该场景族信息容量不绑定,诚实负发现;cut 为可行性区域的理论必要条件)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.15;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 32;`FORMAL_PROOFS.md` 对应定理待补(见 HISTORY §27.8)。

### 1.25 最新: FRIDS 理论形式化与可靠性鲁棒性(2026-08-17,advice/011 驱动,Gate F0-G5)

- **四定理正式化(FORMAL_PROOFS §5B)**: 4.94 可靠信息恒等式(两级独立擦除信道复合,`g ≡ KL(final)`,无 proxy);4.95 原始-对偶严格导出 `J = y·g/D`(需求归一化 LP 强对偶,单纯形出自 z 驻值);4.96 mirror-descent `O(√(T log Q))` 有限时间界(诚实限定瞬时松弛跟踪);4.97 信息负载 cut(ρ>1 ⇒ 必要性不可行,非充分)。
- **F0-G5 可靠性鲁棒性(负结果 + 洞察)**: 全局 κ 失配下 nominal ≈ oracle(仅 +2%,clip 失真);逐链路失配 +1.7%(价格动力学吸收);**均匀 (1−ε) 最坏点缩放是 NO-OP(不改变 argmax)⇒ Robust-FRIDS 无价值,保持 FRIDS-v2**;理论: `g(s)=sI+` 单调线性 ⇒ 区间最坏点封闭,但"是否值得鲁棒"由尺度不变性决定。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.16;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 33;`FORMAL_PROOFS.md` 定理 4.94-4.97 已补。

### 1.26 最新: 可行性包络与瓶颈子集定律(2026-08-17,advice/012 驱动,Gate F0-G6)

- **最强 cut(定理 4.98)**: `ρ* = max_S ρ(S)`,`F(S)` 单调次模 + `D(S)` 模化 ⇒ `λHF−D` 次模 ⇒ **子模最小化(FW 最小范数点)+ 二分**多项式求 ρ*,非枚举;与暴力枚举 Q≤8 逐点一致。
- **相图(K=16,Q/K∈{0.25..2} × s∈{0.95..0.2})**: **27 Green / 3 Yellow(边缘 P_MD 0.071-0.083)/ 0 Red**;`ρ_I* ≤ 0.11`(信息容量从不绑定)、`ρ_C=0.71`(通信预算不绑定)⇒ 三类不可行分离: 唯一活性区域为边缘 coordination-limited(3 格)。
- **可行性利用率 Γ∈[0.75,1.0]**: FRIDS 已贴近可实现边界 ⇒ **停止调度器优化**(advice/012 规则);Q>K 的 owner 循环分配与 token 级联记账为审计基础修复。

### 1.28 最新: P1 证明口径加固(2026-08-22,advice/004 P0.5,Gate P1-hardened)

advice/004 审计 P1 gate 后指出 7 处"理论结构正确但 Gate 未验证到所声称对象"。全部闭合(FRIDS-v2 全程未动):

- **停止鞅 fill-forward**: `simulate_frids_v2(bridge=True)` 在目标停止后对 L/A/V/M/S/r/n_served 保持终值成 `M_{t∧T}`(optional stopping 形式),不再回零。
- **Freedman 口径**: 验证对象改为**联合事件** `P(M_t≤−η, V_t≤v)`(确定性 (η,v) grid,`η` 用方差上界分数、`v` 用 V_up 上界分数),并新增 **time-uniform / line-crossing(Freedman maximal)** 形式 `P(∃t: M_t≤−η, V_t≤v)` ── stopping time 的自然对象。全部 8 场景点态+时一致 0 违例(max_ratio 0.25-0.41,界开始有信息量)。
- **量化 score drift 口径**: 定理改为部署 score 过程 `L̂ = Ã + M`(`Ã` 用部署量化原子 H1 条件漂移,非精确 KL),`δ_Q = |g − g̃|` 单独记录(5-10%);**不再称 A 为 KL(final observable kernel)**。
- **Pathwise stopping tail**: `stopping_tail_verify` 对每条 H1 路径各自算 `(A_q(t)−D_q)_+` 的指数项再求期望(每路径自身 A/V),不再把 mean(A)/mean(V) 塞入非线性界。fill-forward 后停时尾界前提真正激活(117-314 案例/场景),0 违例。
- **Static-MD shadow(定理 4.111 真正关闭)**: 永不停止、固定 D/g、**单一公共 shadow price** + 衰减步长 `μ_t=μ/√t` exp-gradient,T∈{20,40,80,160,320}。**关键发现**: per-UAV local dual 会 herd(全体追同一目标,gap 不收敛 ≈0.16)── 这本身就是 eps_loc 机制;理论对象是公共价格,其 `z* − min_q ṝ_q(T)` 按 `O(√(logQ/T))` 或更快衰减(实测 log-log 斜率 −0.63..−1.10)。
- **eps_loc 重定义**: 边界归一化爆炸(D→0)是量纲伪影;新增 `eps_loc_static_mean`(固定初始缺陷,0.014-0.028)与 **bottleneck `eps_loc`**(local-vs-common CRN 瓶颈目标服务差,≤0.05 全场景,对应 F0-G9A ~1.8% 延迟)。
- **cross-scenario verdict**: 遍历全部 8 场景(4 档 × 2 draw)。
- 注: 门 `results/delay_bridge_gate_p05.json`;测试 `tests/test_reliable_service_bridge.py`(11 项);FORMAL_PROOFS §5E(4.110/4.111)已按加固口径重写。

### 1.27 最新: 服务-时延理论桥(2026-08-22,advice/003 P1,Gate P1)

- **问题**: mirror-descent regret `R_T` 是瞬时松弛的,最终指标是序贯检测时延;缺一句站得住的理论回答"为什么 FRIDS-v2 逼近可行边界"。
- **桥-1(定理 4.110,数值精确)**: 部署规则的 owner LLR 增量的 H1 条件漂移恰为可靠信息 `E_1[Z|F]=Σ x g`(定理 4.94),故 `L_q(t)=L_q(0)+A_q(t)+M_q(t)` 精确成立(`A_q` 累计 predictable reliable service,`M_q` 鞅);有限符号表(量化+BSC+擦除)⇒ 增量可微有界 `b_q`,无需高斯 shortcut,直接 Freedman 型尾界;停时尾界 `P_1(T_q>t) ≤ β_q + exp[-(A_q(t)-D_q)²/(2(V_q(t)+b_q(A-D)/3))]`(服务不足+波动+漏检预算是仅有的失败通道)。**数值**: 分解浮点精确(`~1e-15`),鞅残差零均值,Freedman 不违例(0/480-1280 例)。
- **桥-2(定理 4.111,门通过)**: 需求归一化服务时间平均 `min_q (1/T)Σ r_q(t) ≥ z* - O(√(logQ/T)) - eps_loc`;静态松弛 `z*`(定理 4.95 LP)与时间平均归一化服务之差 `0.02-0.26` ∈ `√(logQ/T)≈0.166 + eps_loc`,桥 2 闭合。
- **诚实边界(advice/003 §9 Case A/B/C)**: 停时尾界的前提 `A_q(t)≥D_q` 在测试工作点很少激活(1-8 例/场景,目标在 ~2.6-4 周期被实现漂移停下)—— 尾界是**结构性充分界**,在中可行区内不紧预测,只在可行性边界(Γ 下降处)才成为绑定解释;**不写一步式 `R_T⇒ΔT`**,两座桥分开陈述(定理 4.110/4.111)。量化修正审计: 5-bit token 的 `A_raw`(未量化 g 记账)与部署原子漂移 `A` 相对差仅 7-12% ⇒ **有限阈值修正太小,FRIDS-v2 保持冻结**(advice/003 §4 规则)。
- 注: 记录/验证模块 `uav_otfs_isac/reliable_service_bridge.py`,桥接记录 `simulate_frids_v2(bridge=True)`(L/A/V/M/S/r/T),门 `scripts/run_delay_bridge_gate.py`(`results/delay_bridge_gate.json`),测试 `tests/test_reliable_service_bridge.py`(8 项)。
- 注: 以上结果记录于本文档与 `SYSTEM_RESEARCH_REPORT.md` §4.17;历史: `THEORY_DEVELOPMENT_HISTORY.md` 阶段 34;`FORMAL_PROOFS.md` 定理 4.98 已补。

## 2. 定理-主题索引(压缩)

| 主题 | 定理/引理 | 模块 |
| --- | --- | --- |
| 融合 | 2.1-2.5 | `fusion` |
| 期望-P_D/次模 | 3.1-3.4 | `expected_pd` |
| RIS/量化/部署 | 4.1-4.6 | `ris_*`, `deployment_search` |
| 精确选择 | 4.7-4.7G, 4.65 | `exact_quota_selection`, `scalable_selection`, `joint_allocation` |
| 孔径/注水 | 4.8-4.13 | `ris_subarray`, `power_split_theory` |
| 系统证书 | 4.14-4.17, 4.24-4.25 | `exact_allocation`, `nomp_refinement` |
| 共识/计数 | 4.18-4.23, 4.40-4.43A | `distributed_consensus`, `theory_guarantees` |
| 信息预算/切换 | 4.44-4.51, 4.56-4.57 | `fundamental_info`, `architecture_switch` |
| 移动/预测 | 4.26, 4.52-4.55 | `stochastic_mobility`, `covariance_aware_ris` |
| 鲁棒分配 | 4.58-4.64 | `robust_portfolio`, `channel_degradation`, `erasure_dominance`, `mobility_envelope`, `physical_link_model` |
| 联合功率-比特/通信模糊 | 4.66-4.73 | `joint_power_bit`, `communication_aware`, `communication_ambiguity`, `robust_joint_power_bit` |
| WTA/反馈 | 4.74-4.78 | `power_split_theory`, `error_feedback` |
| NOMP | 4.79-4.82 | `nomp_refinement` |
| 学习-优化 | 4.83-4.90 | `mappo_nomp_adapter` |
| 检测信息/主动检测 | I± 漂移, 擦除缩放, DP 链, Chernoff, FFT 顺序 P_D | `detection_information` |
| 检测量化/信息梯度 | I+(b) 单调(可证), 凹性否定, 设计度量层级, 1-bit LLR 结构 | `detection_quantization` |
| 精确 max-min 分配 | floor-cover(4.91), NP 可容许性(4.92), 注水 4.7% 界(非定理) | `detection_quantization` |
| Belief-Bellman 主动控制 | 预算状态 Bellman `V_t(pi,B)`(递归+单调+宽松等价), Blackwell LP(一级) + 值界剪枝(二级), 残差 z 标准化与 τ 触发/迟滞, 信息下界 | `active_detection_bellman` |
| 目标对齐序贯检测 | 延迟 Bellman(续观=1 周期), 双阈值数值校准, ν 加权联合 oracle(min-max), 约束内嵌停止 | `active_detection_bellman` |
| 分布式信息审计(F0) | 信息结构隔离(全局/全消息/紧凑 token/零通信), 局部 dual G 值 + 拥塞价格 ψ, 逐模式校准(通信域 belief), 差距分解 Δ_decentral/Δ_comm/Δ_coop, 稳定性指标 | `distributed_audit` |
| 服务-时延理论桥(P1) | LLR=E1 漂移精确分解 `L=A+M`(鞅), Freedman 停时尾界(定理 4.110), 归一化服务时间平均 ≥ z*−O(√(logQ/T))−eps_loc(定理 4.111), 量化修正 7-12% 冻结判定 | `frids (bridge=True)`, `reliable_service_bridge`, `scripts/run_delay_bridge_gate.py` |

## 3. 理论非主张(显式边界)

- 单参数族仅在 `P_D > 0.5` 时证明包含全局线性得分最优;低于该工作点不主张全局最优或单调。
- G18 等为**局部**证书;分配证书限于 `T<=3` 多块;位置证书粒度 0.5m。
- NOMP 贪心 + refine 为启发式,无全局最优性证明(数值对照)。
- 组合精确性不延伸到离散化值预言机与一维融合搜索的容差控制。
- 波束外无 RIS 互耦/极化/波形级响应模型;记账采用保守 1-symbol-per-bit。
- 主动/贪心检测策略为启发式(无全局最优性证明);顺序测试用 `P_D(n) >= 1-beta` 检查点而非 SPRT 双阈值。
- Gate A/B 相关性为经验结果(非因果定理);预算松弛下主动与静态无差异已如实记录。
- `I+(b)` 不满足凹性(证明性否定),信息梯度注水为启发式(界 4.7% 为实测非定理);跨设计 `I+` 排名不可靠(62.5% 失败),Chernoff 亦为代理(14.6% 失败),精确 `P_D` 为基准。
- Gate C Part B 中双侧窗口增益在该物理区间可忽略(0.00017),结构结论与数值增益分离陈述。
- 网格统计量非精确 LLR: 计算 `P_D(b)` 的比特单调性仅对真实 LLR 统计量成立(Fact 4.92),实现违反 ≤ 0.008 由诊断如实报告;floor-cover(定理 4.91)无单调性依赖,精确。
- 信息梯度注水无精确性主张(4.7% 界为实测);精确 max-min `P_D` 分配为定理 4.91。
- **Gate D1 判定为非主张的负结果(如实)**: 单目标预算 Bellman 相对一步贝叶斯前瞻无实质增益(≤2.7%),故 belief-state Actor-Critic 蒸馏(advice §1-2 第二步)按决策规则**暂缓**;该判定仅覆盖 Q=1 代价目标框架,不排除 Q≥2 或 P_D 阈值目标下的价值化收益。
- **残差统计量的前提与局限**: `tau = min(|mean_0|,|mean_1|)` 的"正确模型 → 0"性质要求单假设流(评估协议逐假设进行);混合 H0/H1 流下两条件均值均非零,需另行校准。flip-only 失配不触发(LLR 原子幅度不变),仅 success/擦除类失配被检测。
- **分配-时间闭环负结果**: 预算 Bellman G 值调度在 Gate B 的 P_D 阈值框架下跑输 τ_pred(最差 T 3.50 vs 2.08),价值函数的代价目标与 P_D 目标不一致,不构成对闭环思想的否定性定理,只构成对该实例化的否定。
- **值界剪枝(二级)为精确消去**: 保守化(同预算层)保证不误剪;但消去是"当步"的,跨步可能重新有用(实现每步重评估),无全局动作约简主张。
- **Gate D2 判定边界(如实)**: D2-A 负结论仅覆盖 Q=1 单目标;D2-B 正结论(+14.1%)在 Q=2 强-弱竞争、共享预算、min-max 度量下成立,误差均在 α=0.05 附近(weak 目标 P_MD 0.077-0.085 略超,共享预算资源限制,所有策略同等);ν 扫描为 min-max 的松弛实现(有限网格),非连续对偶收敛证明;`T_q` 的双阈值停止为数值校准(非闭式),校准网格与 MC 样本决定精度。
- **F0 审计边界(如实,advice/005)**: 审计结论限于单一场景类(K=6,Q=3,随机链路 + 弱目标,固定 owner,R_coord ≤ 2,5-bit token 证据,单决策规则);"5-bit 量化损失小(+1.8%…+6.3%)"为该核集与 token 布局下的实测,不构成一般性定理;token bits 可行性依赖场景(4-bit 在测试类不可行,为 infeasible region 实证而非定理);Δ_comm 的方向(C ≥ B)与信息论一致,但幅度受校准种子影响(近并列可行边界的 MC 选择,敏感性已写入 JSON);local-only 基线在 40 周期截尾,其 P_MD 为截尾实测;duplicate sensing 率 ~1.0 由 min-max 瓶颈聚焦引起,是稳定性指标的语义边界(非失稳证据);D_L 按单假设流报告,混合流语义未校准;决策规则(dual G 值 + 线性 ψ)为启发式,无全局最优性证明,F2 将与其他协调算法同信息/同预算公平对比。
- **重分类声明(advice §10)**: Exact/联合 oracle 定位为离线审计与结构验证;Bellman/dual/rollout 定位为局部动作价值层;NOMP/MAPPO 不再预先绑定身份,F2 公平对比后决定去留(现仍记录其历史结果,不删除)。
- **F0-S 规模审计边界(如实,advice/006)**: 每档单一场景抽样(冻结 scenario_seed=0),J 的 ± 为模拟种子 SE,不覆盖场景抽样方差;J 非单调((8,4) 最优)部分来自弱目标占比随 Q 下降(仅 q=0 为弱);P_MD 判定在 2pp 余量处敏感((12,6) 0.070 vs β+2pp=0.07);Gate A 的 +13.1% 为该场景族内 16-vs-6 的实测增长(约 5 SE),非一般性定理;rx 负载线性为构造性(全 mesh 每 UAV 收 K−1 token),不构成"必须做稀疏拓扑"的结论 —— 仅当它先于检测层成为约束时;下一步方向由本审计结果决定(target allocation / resource competition),不再预先承诺 F1/F2/稀疏/owner 等方向。
- **F0-A 分配审计边界(如实,advice/007)**: Case 判定基于 4 档单场景的配对诊断,签名阈值(±10%/±30%)为约定非定理;ρ_alloc 为逐周期合并 Spearman(仅未决目标,含秩并列);regret 的 oracle 参考使用同一 biased 索引 —— 它度量**协调损失**,测不出索引偏置本身(偏置由增益结构诊断暴露: 弱目标 −1.0 vs 易目标 2.5e7);归一化索引修复在 (12,6) 有效(−11%)而 (16,8) 无效,后者的 worst-target 由单个内在困难目标决定 —— 不构成"归一化恒有益"的一般性主张;age/价格参数(η_A、η、γ)的取值仅为测试网格,未做全局调优;修复实验未改变冻结的校准/阈值/token 协议。