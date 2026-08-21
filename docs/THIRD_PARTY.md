# GitHub implementation survey and reuse boundaries

Survey date: 2026-08-02.

## Candidate repositories

### whatshow/Phy_Mod_OTFS

- URL: https://github.com/whatshow/Phy_Mod_OTFS
- Relevance: Python/MATLAB OTFS modulation, fractional Doppler, channel paths,
  DD-domain channel matrices, modulation and demodulation.
- License observed in checkout: GNU GPL v3, plus an additional `ANTI 996
  LICENSE` file.
- Decision: **optional external backend only**. The project does not copy or
  modify its source. Redistribution of a combined work requires an independent
  license review and compliance with the upstream terms.

### YongzhiWu/OTFS_radar

- URL: https://github.com/YongzhiWu/OTFS_radar
- Relevance: MATLAB OTFS ISAC waveform generation, coarse/fine delay-Doppler
  search, matched-profile sensing.
- License observed: no license file in the cloned repository.
- Decision: **algorithmic reference only**. No code was copied because public
  source without a license is not automatically reusable.

### lkk688/AIsensing

- URL: https://github.com/lkk688/AIsensing
- Relevance reported by the search index: reusable radar DSP and ISAC modules
  including OFDM/OTFS/FMCW.
- Checkout status: the shallow clone did not complete into a valid `HEAD` in
  this environment, so code and license could not be verified.
- Decision: not used.

### acyiobs/sensing_aided_OTFS_channel_estimation_

- URL: https://github.com/acyiobs/sensing_aided_OTFS_channel_estimation_
- Relevance: sensing-aided sparse OTFS channel estimation.
- License reported by the public index: CC BY-NC-SA 4.0.
- Decision: not used because it is non-commercial/share-alike and does not
  directly implement the distributed detection/selection problem.

## Original implementation in this repository

The following components were implemented independently from the equations in
the supplied document: exact/SAA expected deflection, Schur-complement marginal
gain, covariance regularization, BSC quantization moments, detectable-erasure
sets, two-stage QoS-aware selection, fusion, and Monte Carlo detection.

The later receiver-chain components are also independent implementations:
toy MF/CFAR front end with sidelobe-aware CFAR and noncoherent integration,
per-path Fisher-type covariance from matched-filter curvature, multistatic
association gates, evidence-moment calibration (G1-A), report-channel closure
(G1-B), conditional-ranking and greedy-vs-Oracle gates (G1-C/D), and the
system-level sweep (G2).  These use only the repository's own modules and the
declared dependencies (NumPy, SciPy, PyYAML, pytest); no third-party source
code is copied.
