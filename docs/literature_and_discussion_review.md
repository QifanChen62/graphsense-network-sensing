# Literature and Discussion Review

## Sources Reviewed

Official sources:

- GraphChallenge challenges page: https://graphchallenge.mit.edu/challenges
- GraphChallenge data sets page: https://graphchallenge.mit.edu/data-sets
- GraphChallenge submit page: https://graphchallenge.mit.edu/submit
- GraphChallenge champions page: https://graphchallenge.mit.edu/champions
- Anonymized Network Sensing paper: https://arxiv.org/abs/2409.08115

Reference/data search:

- Searched for CAIDA/ILANDS-sensor, `pcap2grb`, GraphBLAS traffic-matrix examples, GitHub issues, Reddit/forum discussion, and small official samples.
- Local workspace search found no official reference checkout, `pcap2grb` tree, ILANDS directory, `.pcap` file, or `.grb` file.
- Public search did not surface a small no-friction official sample suitable for a default repo test. The official data paths appear to include multi-GB PCAP/GraphBLAS assets or access-controlled CAIDA material.

Related methods:

- Count-Min Sketch for streaming approximate counts.
- Misra-Gries/space-saving style heavy-hitter tracking.
- Recent GraphChallenge reference-implementation improvement work.
- Sketching data streams for network measurement.
- Traffic feature-distribution anomaly analysis.
- Sparse matrix traffic analysis, GraphBLAS-style graph analytics, traffic-matrix heavy hitters, entropy/concentration summaries, and scanner/fanout anomaly patterns.

## Official Requirements That Matter

- The GraphChallenge is a paper-style submission, not a leaderboard-only contest.
- The submit page allows teams to choose challenge elements, data sets, and metrics that best demonstrate the innovation.
- The Anonymized Network Sensing track centers on transforming packet/flow-like records into sparse source-destination traffic matrices and computing sensing quantities from those matrices.
- The data page lists large official PCAP/GraphBLAS/CAIDA resources. This makes a lightweight default workflow valuable, but it also means this project must not claim official full-scale results unless those assets are actually used.

## What Prior Winners Emphasize

The champions page shows that top GraphChallenge entries often emphasize high-performance systems, GPUs, C++/CUDA, RAPIDS, GraphBLAS, database/system integration, and large-scale benchmarking. Those directions are credible but difficult to reproduce for many student entrants on commodity laptops.

## Remaining Gap For Student Innovation

The defensible student gap is not raw speed. It is a reproducible and inspectable path for learning and testing network-sensing ideas:

- deterministic synthetic regimes with controlled sparsity, duplication, skew, community structure, scanner/fanout behavior, and temporal bursts;
- transparent exact baselines that agree on the constructed sparse matrix;
- a streaming/sketch path that can be studied as an approximation method;
- figures and CSVs generated from scripts;
- honest documentation of where approximation succeeds and fails.

## Discussion/Issue Pain Points

No strong public Reddit/forum thread or actionable GitHub issue was found that provided a ready-made small official benchmark path. The practical pain point inferred from official data organization and reference-code framing is setup friction: official data can be large, reference paths may require separate checkout/build steps, and students need a smaller reproducible entry point before attempting full data.

## Chosen Innovation Direction

Chosen thesis:

**Shape-Aware Sparse and Streaming Analytics for Reproducible Anonymized Network Sensing.**

Why this is stronger than only synthetic speed benchmarks:

- It adds a genuinely different algorithmic mode: streaming/sketch analytics with bounded sketch memory.
- It turns negative results into evidence: the same sketch recovers hotspot heavy hitters but fails on diffuse scanner/uniform traffic.
- It adds a sketch-width sensitivity study: larger Count-Min Sketch widths reduce hotspot weight error but do not solve diffuse-regime candidate discovery at fixed candidate capacity.
- It adds a candidate-capacity study: diffuse-regime recall improves only when many more candidate pairs are retained, which makes the approximation tradeoff explicit.
- It adds an online method selector: early-stream duplication, nonzero growth, entropy, Gini, and top-share features recommend sparse direct or pandas groupby and are checked against saved exact-method benchmark winners.
- It adds time-bin anomaly demonstrations using total traffic, fanout, destination concentration, distributed low-rate scan coverage, and robust positive-deviation scores.
- It makes traffic shape central to method choice, which is scientifically more useful than claiming one method wins everywhere.
- It validates on real challenge data: a 5,000,000-packet byte-range prefix of the official anonymized capture is fetched and parsed by a small reproducible script, and the selector, certificate, and capacity findings are checked against it.
- It remains honest: experiments are synthetic commodity-CPU benchmarks plus an official-capture prefix, the reference comparison is local paper-formula code, and no official full-reference or full-capture-scale result is claimed.

Positioning against reference-code improvement work:

- Reference-implementation improvement papers are closest in spirit because they improve official GraphChallenge code paths.
- This project is positioned differently: it is a student-facing CPU toolkit with deterministic synthetic regimes, explicit approximation failure diagnostics, temporal sensing summaries, and an online selector.
- The paper should avoid implying it replaces official high-performance reference work.
