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
- [x] Literature and discussion review saved in `docs/literature_and_discussion_review.md`.
- [x] IEEE/HPEC-style paper draft under six pages.
- [x] Paper compiles to a PDF under the six-page limit with the current draft content.
- [x] Current repo hygiene audit found no huge files, raw PCAP/GRB data, private keys, or credentials.
- [x] Local official-reference/data search found no available checkout or small official sample in the workspace.

## Optional If Official Data Is Available

- [ ] Run the full official GraphChallenge reference code on the same small input if a reference checkout and compatible input path are available.
- [x] Replace or supplement synthetic benchmarks with a small official/example data slice if access is available. (Done via the 256 MiB byte-range prefix of the official capture.)
- [ ] Add a true official-reference comparison table only after actually running the official reference implementation.

## Must Do Before Upload

- [x] Replace TODO author metadata in `paper/main.tex` with final author name, institution, and email.
- [ ] Recompile `paper/main.tex` and confirm the PDF remains at most six pages.
- [ ] Audit all claims: no Champion-level, full CAIDA-scale, PCAP parse-time, official-data, or official-reference performance claims unless directly measured.
- [ ] Confirm the official-reference/data caveat is still visible in `README.md`, `docs/challenge_notes.md`, and the paper.
- [ ] Audit the repo for huge raw data, PCAPs, credentials, and unintended machine-local paths.
- [ ] Confirm bibliography formatting and final submission metadata.
