# Related Work

Draft literature review for the paper.  Citations are draft-level; verify
venues, volume/page numbers, and author lists before submission.

## 1. OTFS-based integrated sensing and communication

OTFS has become a standard waveform for high-mobility ISAC because the
delay-Doppler (DD) domain concentrates doubly selective channels and target
returns.  Representative works:

- R. Hadani et al., "Orthogonal time frequency space modulation," IEEE WCNC,
  2017, DOI: 10.1109/WCNC.2017.7925924.
- P. Raviteja, K. T. Phan, Y. Hong, and E. Viterbo, "Interference
  cancellation and iterative detection for orthogonal time frequency space
  modulation," IEEE Transactions on Wireless Communications, vol. 17,
  no. 10, pp. 6501-6515, 2018.
- State-of-the-art review of OTFS for integrated radar-communication,
  including radar receiver and MIMO waveform design
  ([TUDelft repository](https://repository.tudelft.nl/record/uuid:42d52924-2efc-4d34-92a9-1eeadc0baec4)).
- Unified variational-inference receiver for OTFS-based MIMO ISAC
  ([Beijing Institute of Technology](https://pure.bit.edu.cn/en/publications/integrated-sensing-and-communication-receiver-design-for-otfs-bas/)).
- Sparse target parameter and channel estimation in mmWave MIMO OTFS-aided
  ISAC ([NSTL](https://cd.nstl.gov.cn/paper_detail.html?id=92baa18d5bb91c4238e9384bf5a86e84)).
- RFSoC-based scalable OTFS prototyping platform for ISAC
  ([IEEE](https://ieeexplore.ieee.org/document/10974457)).

Gap: these works optimize waveform/receiver processing, not the
post-communication selective fusion of correlated soft evidence under a
finite report bit budget.

## 2. RIS-assisted ISAC and RIS-assisted OTFS

RIS creates controllable propagation paths and is a 6G enabler for ISAC.
Representative works:

- Q. Wu and R. Zhang, "Intelligent reflecting surface enhanced wireless
  network via joint active and passive beamforming," IEEE Transactions on
  Wireless Communications, vol. 18, no. 11, pp. 5394-5409, Nov. 2019.
- RIS-enabled ISAC from theory to practice
  ([IEEE](https://ieeexplore.ieee.org/abstract/document/11475735)).
- Beamforming design for RIS-assisted mobile ISAC based on OTFS
  ([IEEE](https://ieeexplore.ieee.org/document/11447917/references)).
- Survey of RIS-assisted OTFS systems, including RIS phase design and
  RIS-assisted ISAC
  ([Semantic Scholar](https://www.semanticscholar.org/paper/A-Survey-on-Reconfigurable-Intelligent-Orthogonal-Tao-Li/b0aca4a69cc7758843b0a06f263302943428c8bb)).

Gap: prior work treats RIS phase as a beamforming variable for waveform or
link quality; our work jointly couples RIS phase/placement, phase-resolution
control bits, and report-selection bits under one total budget, with a
monotone expected-P_D objective.

## 3. UAV-ISAC

UAVs add mobility and deployment flexibility to ISAC.  Representative works:

- Comprehensive survey of UAV-aided ISAC for 6G
  ([IEEE](https://ieeexplore.ieee.org/document/11333384/similar)).
- Low-altitude UAV swarm ISAC: new opportunities and challenges
  ([EITEE](https://www.academax.com/EITEE/doi/10.1631/ENG.ITEE.2026.0030)).

Gap: UAV-ISAC surveys emphasize trajectory, beamforming, and bandwidth, but
do not model the selective soft-information reporting chain after
quantization, BSC, and correlated erasures.

## 4. Distributed detection and soft-information fusion

Distributed detection with quantized soft decisions is a classical problem.
Representative works:

- R. R. Tenney and N. R. Sandell, "Detection with distributed sensors,"
  IEEE Transactions on Aerospace and Electronic Systems, vol. AES-17, no. 4,
  pp. 501-510, July 1981.
- Z. Chair and P. K. Varshney, "Optimal data fusion in multiple sensor
  detection systems," IEEE Transactions on Aerospace and Electronic Systems,
  vol. AES-22, no. 1, pp. 98-101, Jan. 1986.
- P. K. Varshney, "Distributed Detection and Data Fusion," Springer, 1997.
- Y. H. Wang, "On the number of successes in independent trials," Statistica
  Sinica, vol. 3, no. 2, pp. 295-312, 1993.
- R. Olfati-Saber and R. M. Murray, "Consensus problems in networks of
  agents with switching topology and time-delays," IEEE Transactions on
  Automatic Control, vol. 49, no. 9, pp. 1520-1533, Sep. 2004.
- Quantized fusion rules for energy-based distributed detection in
  bandwidth-constrained sensor networks
  ([arXiv](https://ar5iv.labs.arxiv.org/html/1506.01210)).
- Cross-layer resource allocation for distributed detection with soft
  decision fusion
  ([IEEE](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000005156517)).

Gap: these works assume independent or i.i.d. observations and treat
quantization/BSC as per-sensor noise; our system explicitly models
correlated common-state erasures, moment-matched heteroscedastic evidence,
and set-dependent fusion ranking.

## 5. Communication-constrained resource allocation in ISAC

Resource competition between sensing and communication is a core ISAC
problem.  Representative works:

- F. Liu, C. Masouros, A. P. Petropulu, H. Griffiths, and L. Hanzo, "Joint
  radar and communication design: Applications, state-of-the-art, and the
  road ahead," IEEE Transactions on Communications, vol. 68, no. 6,
  pp. 3834-3862, 2020, DOI: 10.1109/TCOMM.2020.2973976.
- P. Kumari, J. Choi, N. Gonzalez-Prelcic, and R. W. Heath, "IEEE 802.11ad-
  based radar: An approach to joint vehicular communication-radar system,"
  IEEE Transactions on Vehicular Technology, vol. 67, no. 4, pp. 3012-3027,
  2018.
- A. Liu et al., "A survey on fundamental limits of integrated sensing and
  communication," IEEE Communications Surveys & Tutorials, vol. 24, no. 2,
  pp. 994-1034, 2022, DOI: 10.1109/COMST.2022.3149272.
- Joint sparse resource allocation with one-bit receivers for ISAC
  ([IEEE](https://ieeexplore.ieee.org/document/11443938/references)).
- Performance analysis of ISAC by non-orthogonal multiplexing
  ([IEEE Communications Letters](https://www.x-mol.com/paper/1724848142568673280)).

Gap: prior allocation schemes split time/frequency/power between radar and
communication or optimize beamforming; they do not allocate per-report
quantization bits and RIS control bits jointly under the expected-P_D
objective with a provable monotone/submodular structure.

## 6. Positioning of this paper

The paper fills the intersection of the five lines:

- Scenario: RIS-assisted UAV-OTFS-ISAC with a direct-plus-cascaded sensing
  channel (6G ISAC).
- Channel: physics-based two-way radar and three-leg cascaded path loss,
  phase quantization, and control-overhead accounting.
- Reporting: quantization, BSC, correlated erasures, and exact reception
  patterns.
- Fusion: KKT-derived monotone `P_D`-optimal linear family (G3).
- Selection: expected-`P_D`-gain greedy with bounded-regime submodularity
  (G4), joint RIS phase/placement/report/control allocation (G5), and
  Lipschitz-certified deployment search (G5-U/V/W).

To the best of the draft's knowledge, no prior work combines all of these
elements into one audited, reproducible chain.

## 7. Reference list (draft, verify before submission)

- R. Hadani, S. Rakib, M. Tsatsanis, A. Monk, A. J. Goldsmith, A. F. Molisch,
  and R. Calderbank, "Orthogonal time frequency space modulation," in Proc.
  IEEE Wireless Communications and Networking Conference (WCNC), 2017.
- P. Raviteja, K. T. Phan, Y. Hong, and E. Viterbo, "Interference
  cancellation and iterative detection for orthogonal time frequency space
  modulation," IEEE Transactions on Wireless Communications, vol. 17,
  no. 10, pp. 6501-6515, 2018.
- W. Yuan, L. Zhou, S. K. Dehkordi, et al., "From OTFS to DD-ISAC:
  Integrating sensing and communications in the delay Doppler domain,"
  arXiv:2311.15215, 2023. [arXiv](https://arxiv.org/abs/2311.15215)
- Q. Wu and R. Zhang, "Intelligent reflecting surface enhanced wireless
  network via joint active and passive beamforming," IEEE Transactions on
  Wireless Communications, vol. 18, no. 11, pp. 5394-5409, 2019.
- Y. Xu et al., "Joint beamforming for RIS-assisted integrated sensing and
  communication systems," IEEE Transactions on Communications, vol. 72,
  no. 4, 2024.
- K. Meng et al., "UAV-enabled integrated sensing and communication:
  Opportunities and challenges," IEEE Wireless Communications, vol. 31,
  no. 2, pp. 97-104, 2024.
- E. Nurellari, S. A. Aldalahmeh, M. Ghogho, and D. C. McLernon, "Quantized
  fusion rules for energy-based distributed detection in wireless sensor
  networks," arXiv:1506.01210, 2015.
  [arXiv](https://arxiv.org/abs/1506.01210)
- S. Zargari, D. L. Galappaththige, C. Tellambura, and H. V. Poor, "A
  Riemannian manifold approach to constrained resource allocation in ISAC,"
  IEEE Transactions on Communications, 2024,
  DOI: 10.1109/TCOMM.2024.3487801.
- A. Liu, Z. Huang, M. Li, Y. Wan, W. Li, T. X. Han, C. Liu, R. Du, D. K. P.
  Tan, and J. Lu, "A survey on fundamental limits of integrated sensing and
  communication," IEEE Communications Surveys & Tutorials, vol. 24, no. 2,
  pp. 994-1034, 2022, DOI: 10.1109/COMST.2022.3149272.
- F. Liu, C. Masouros, A. P. Petropulu, H. Griffiths, and L. Hanzo, "Joint
  radar and communication design: Applications, state-of-the-art, and the
  road ahead," IEEE Transactions on Communications, vol. 68, no. 6,
  pp. 3834-3862, 2020, DOI: 10.1109/TCOMM.2020.2973976.
- P. Kumari, J. Choi, N. Gonzalez-Prelcic, and R. W. Heath, "IEEE 802.11ad-
  based radar: An approach to joint vehicular communication-radar system,"
  IEEE Transactions on Vehicular Technology, vol. 67, no. 4, pp. 3012-3027,
  2018.
- R. R. Tenney and N. R. Sandell, "Detection with distributed sensors," IEEE
  Transactions on Aerospace and Electronic Systems, vol. AES-17, no. 4,
  pp. 501-510, 1981.
- Z. Chair and P. K. Varshney, "Optimal data fusion in multiple sensor
  detection systems," IEEE Transactions on Aerospace and Electronic Systems,
  vol. AES-22, no. 1, pp. 98-101, 1986.
- P. K. Varshney, "Distributed Detection and Data Fusion," New York, NY,
  USA: Springer, 1997.
- Y. H. Wang, "On the number of successes in independent trials," Statistica
  Sinica, vol. 3, no. 2, pp. 295-312, 1993.
- R. Olfati-Saber and R. M. Murray, "Consensus problems in networks of
  agents with switching topology and time-delays," IEEE Transactions on
  Automatic Control, vol. 49, no. 9, pp. 1520-1533, 2004.

## 8. Algorithm evolution

The proposed chain is positioned as an evolution, not a disconnected set of
gates:

1. Classical distributed detection derives person-by-person optimal local
   rules and Chair-Varshney likelihood-ratio fusion for independent sensors
   (Tenney/Sandell; Chair/Varshney; Varshney's book).  Quantized fusion
   under bandwidth constraints extends this line (Nurellari et al.).  The
   gap is correlated common-state erasures and set-dependent fusion ranking.
2. Exact counting/Poisson-binomial theory (Wang) and consensus algorithms
   (Olfati-Saber/Murray) provide the distributed branch; the gap is that
   feasibility is non-monotone in the voter count, so an exact prefix
   certificate is needed instead of a Gaussian or binary-search shortcut.
3. ISAC resource allocation moves to joint beamforming/bandwidth/power under
   sensing constraints (Liu et al.; Kumari et al.; Xu et al.; Zargari et
   al.).  The gap is per-report quantization bits and RIS control bits under
   an expected-`P_D` objective.
4. OTFS/DD-domain ISAC provides the doubly selective sensing waveform and
   DD-resolution interpretation (Hadani; Raviteja; Yuan et al.).  The gap is
   the post-communication fusion chain.
5. UAV-ISAC introduces deployment and mobility as system dimensions (Meng
   et al.).  The gap is the selective soft-information reporting chain.
6. This paper keeps each prior element as a baseline or a degenerate case:
   deflection-optimal linear fusion -> KKT `P_D`-optimal family (G3) ->
   expected-`P_D` greedy (G4) -> exact budget/max-min selection
   (G8-K/G8-M) -> scaled threshold search (G8-S) -> RIS phase/placement,
   variable-rate, and architecture-switch gates (G5-G54).

Each step is audited against the previous one, so the paper can claim an
algorithmic evolution with formal statements rather than a collection of
isolated heuristics.
