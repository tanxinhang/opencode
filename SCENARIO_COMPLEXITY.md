# Scenario Complexity Audit and Upgrade Roadmap

## 1. Current scenario: deliberately simple

The audited mainline uses:

- static UAV, target, receiver, and RIS geometry;
- one RIS with a 1-D uniform linear array;
- scalar path-loss physics, no mutual coupling or polarization;
- fixed 5-bit soft reports or 1-bit hard decisions;
- independent/common/grouped erasure models, no dynamic topology;
- one fusion center or a simplified peer majority;
- moment-matched Gaussian evidence instead of a bandwidth-consistent OTFS SDR;
- fixed total budget and one QoS threshold.

This simplicity is not accidental.  It is what allows the closed-form
`N*` derivation, the max-min water-filling allocation, and the exact
single/multi-block local certificates.  The paper should state this trade-off
explicitly.

## 2. What is already beyond "toy" level

- RIS aperture and control-overhead accounting is exact under the model.
- Subarray multi-beam and joint placement-allocation are optimized on the
  exact system objective with local certificates.
- Distributed consensus is evaluated under partial observability, multi-hop
  erasure, and correlated common failure.
- Scalability is tested at `Q=2/4/6` and `M/Q=1/2/3`.
- Stochastic mobility with AR(1) trajectories, random blockage, and RIS
  one-frame reconfiguration latency is covered by G51.
- Multi-step AR(1) MMSE prediction under RIS latency is covered by G53.

## 3. What is still too simple for a 6G paper

### 3.1 Static geometry and no mobility

The paper cannot claim a 6G UAV scenario without at least one mobility
extension:

- time-varying UAV/target positions;
- time-varying blockage;
- RIS phase reconfiguration latency/cost;
- trajectory coupling with the report budget.

G26 now covers deterministic mobility and time-varying blockage; G51
upgrades this to AR(1) stochastic trajectories, random blockage, and
one-frame RIS reconfiguration latency, so the static-geometry limitation is
partially closed.  G52 adds AR(1) conditional-mean RIS prediction;
continuous-time trajectories, multi-step prediction (G53), and
covariance-aware stochastic-optimal RIS control remain open; G54 reports a
negative result for the expected-gain surrogate under quantization.

### 3.2 Single RIS and 1-D ULA

Current architecture is one 1-D aperture.  A 6G scene should eventually
include:

- 2-D RIS aperture and elevation steering;
- multiple RISs with joint placement;
- mutual coupling and polarization as sensitivity cases.

### 3.3 Waveform-level OTFS is missing

The mainline uses moment-matched Gaussian evidence.  The paper should either
keep this as an abstraction or connect it to a bandwidth-consistent OTFS
front end, otherwise the "OTFS" in the title is only a scenario label.

### 3.4 Reporting layer is too regular

Current reports are fixed-length.  Real ISAC links use:

- variable quantization levels;
- hybrid soft/hard reporting;
- unequal report costs;
- multi-hop routing with dynamic topology;
- network coding or partial reliability classes.

### 3.5 No interference or clutter

The detection metrics assume aligned candidates and no clutter.  A 6G ISAC
paper should add at least one controlled interference/clutter sensitivity.

## 4. Complexity upgrade ladder

The upgrades should be incremental, one per gate, so each keeps a formal
statement:

1. Mobility and time-varying blockage (G26 deterministic, G51 stochastic).
2. Multi-RIS joint placement (G27).
3. 2-D aperture steering (G28).
4. Variable-rate hybrid soft/hard reports (G29).
5. Interference/clutter sensitivity (G30).
6. Bandwidth-consistent OTFS front end (G31).

Each gate should preserve:

- the resource identity;
- exact or certified evaluation;
- communication/sensing principles;
- an explicit statement of what is and is not modeled.

## 5. Recommended first upgrade

The highest value with the least risk is **mobility and time-varying
blockage**.  It can be modeled without replacing the moment-matched
abstraction:

- sample positions from a deterministic trajectory;
- let the weak-target blockage vary over time;
- amortize RIS reconfiguration over the coherence block;
- report the resulting worst-over-time QoS.

This gives a dynamic 6G scene while retaining the closed-form
architecture derivation as the static design layer.
