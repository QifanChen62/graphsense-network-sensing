# Submission Checklist

## Complete

- [x] Reproducible local pipeline with setup check.
- [x] Synthetic/sample data path that does not require protected or multi-GB downloads.
- [x] Sparse traffic-matrix construction and analytics.
- [x] Multi-regime synthetic benchmark covering hotspot, bursty community, scanner fanout, and uniform sparse regimes.
- [x] Controlled synthetic generator supports target nnz/density, duplication rate, skew, community structure, scanner/fanout behavior, and temporal bursts.
- [x] Large benchmark to 1,000,000 and 5,000,000 input edges with different nnz/density profiles.
- [x] Streaming/sketch benchmark compares exact sparse analytics with Count-Min Sketch heavy-hitter approximation.
- [x] Streaming/sketch sensitivity study varies Count-Min Sketch width and saves CSV/figure outputs.
- [x] Candidate-capacity sensitivity study saves CSV/figure outputs and tests candidate-discovery failure.
- [x] Early-stream method selector saves CSV/figure outputs and matches saved 5,000,000-edge exact-method winners on the current synthetic regimes.
- [x] Safety-certified selector saves `results_v1/certified_selector_summary.csv` from a 50,000-record prefix and reports conservative sketch safety decisions.
- [x] Time-bin anomaly analytics saves per-bin CSV summaries and interpretable detection curves for volume burst, scanner fanout, concentrated-destination, and distributed low-rate scan scenarios.
- [x] Benchmark CSVs report construction time, analytics time, edges per second, nnz per edge, and coarse RSS per edge.
- [x] Official-format smoke test round-trips a Matrix Market sparse-coordinate matrix and checks shape, nnz, density, and total weight.
- [x] Local baseline comparison against pandas groupby and Python Counter aggregation.
- [x] Local GraphChallenge-paper-style reference formula compatibility check on a small reproducible sample.
- [x] Benchmark CSV output.
- [x] Figure generation from saved CSVs.
- [x] Official-data prefix validation: `scripts/fetch_official_prefix.py` Range-downloads 256 MiB of the official `pcap.zst`, parses 5,000,000 packets in under five seconds, and commits a provenance manifest.
- [x] Real-data experiment validates the early-stream selector against the measured exact-method winner and sweeps candidate capacity on the official prefix (`results/real_data_summary.csv`, `results/real_data_candidate_sensitivity.csv`).
- [x] Certified selector positive case: width 16,384 and candidate capacity 5,000 certify the hotspot regime (`results_v1/certified_selector_tuned_summary.csv`); diffuse regimes and the official prefix remain correctly uncertified.
- [x] Candidate tracker eviction uses a lazy-deletion heap so all-unique streams cost O(log capacity) per record.
- [x] Segment stability: eight disjoint 5,000,000-packet official-product segments all yield the same selector recommendation and screen refusal (`results/real_data_window_stability.csv`).
- [x] Exact cross-validation against official GraphBLAS matrices: 64/64 windows, 8,388,608 pairs (`results/grb_cross_validation.csv`).
- [x] SuiteSparse:GraphBLAS construction baseline including native 2^32 hypersparse mode (`results/graphblas_comparison.csv`).
- [x] Safety-screen c_min uses the rigorous SpaceSaving retention bound (W/w_k), which quantitatively predicts the empirical capacity sweep.
- [x] Paper language audited: S3 products named as the official synthetic Random PCAP / Random GraphBLAS variants; no real-telescope-traffic claim.
- [x] Literature and discussion review saved in `docs/literature_and_discussion_review.md`.
- [x] IEEE/HPEC-style paper draft under six pages.
- [x] Paper compiles to a PDF under the six-page limit with the current draft content.
- [x] Current repo hygiene audit found no huge files, raw PCAP/GRB data, private keys, or credentials.
- [x] Local official-reference/data search found no available checkout or small official sample in the workspace.

## Optional If Official Data Is Available

- [ ] Run the full official GraphChallenge reference code on the same small input if a reference checkout and compatible input path are available.
- [x] Replace or supplement synthetic benchmarks with a small official/example data slice if access is available. (Done via the 256 MiB byte-range prefix of the official capture.)
- [x] Add a true official-reference comparison at the artifact level: exact match against the official GraphBLAS matrices (64/64 windows). Running the official reference *code* on the same machine remains future work and is labeled as such in the paper.

## Must Do Before Upload

- [x] Replace TODO author metadata in `paper/main.tex` with final author name, institution, and email.
- [x] Recompile `paper/main.tex` and confirm the PDF remains at most six pages. (Six pages exactly.)
- [x] Audit all claims: no Champion-level or full CAIDA-scale claims; official-data claims are limited to the measured 256 MiB / 5,000,000-packet prefix; the official-prefix fetch/parse time is directly measured and recorded in the manifest.
- [x] Confirm the official-reference/data caveat is still visible in `README.md`, `docs/challenge_notes.md`, and the paper Limitations section.
- [x] Audit the repo for huge raw data, PCAPs, credentials, and unintended machine-local paths. (No tracked file over 5 MB; the 256 MiB `.part` cache and parsed CSV are gitignored; v0_stable LaTeX build logs untracked.)
- [ ] Confirm bibliography formatting and final submission metadata.
- [ ] Final PDF read-through before upload.
- [x] Replace the `USERNAME` placeholder in the paper's Artifact Availability URL (paper/main.tex) with the real public repository URL, then recompile: https://github.com/QifanChen62/graphsense-network-sensing
