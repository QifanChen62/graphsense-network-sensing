# Challenge Notes

This repository targets a Student Innovation style submission for the MIT/IEEE/Amazon GraphChallenge Anonymized Network Sensing track.

## Official Sources Checked

- Challenge list: https://graphchallenge.mit.edu/challenges
- Data sets: https://graphchallenge.mit.edu/data-sets
- Submission instructions: https://graphchallenge.mit.edu/submit
- Track paper: https://arxiv.org/abs/2409.08115

## Official Reference/Data Path

The public GraphChallenge data page lists Anonymized Network Sensing resources including a random PCAP archive, a random GraphBLAS-format archive, and CAIDA PCAP access links. The full archives are 8-14 GB, so they are not downloaded in full. Instead, `scripts/fetch_official_prefix.py` Range-downloads the first 256 MiB of the official `pcap.zst` from the public `graphchallenge` S3 bucket, stream-decompresses the truncated zstd frame, and parses 5,000,000 fixed-size packet records into an anonymized `src,dst,bytes` edge table with a committed provenance manifest. This official-capture prefix is used to validate the pcap parser, the early-stream method selector, the certified sketch checks, and the candidate-capacity findings on real challenge data.

The arXiv track paper describes the sparse traffic-matrix construction and Table-I-style graph quantities. This repository therefore provides two levels of reference comparison:

- `scripts/run_reference.py` for a separately checked-out official reference implementation.
- `scripts/official_format_smoke.py` for a small Matrix Market sparse-coordinate round trip that checks GraphBLAS-style matrix compatibility. This is a format smoke test, not an official-data result.
- `scripts/compare_reference.py` for a small local GraphChallenge-paper-style formula baseline that computes traffic matrix counts plus source/destination similarity nnz. This is not the full official reference repository.

## Local Official Data Search

A local search of the repository and nearby project folders found no official GraphChallenge reference checkout, `pcap2grb` tree, ILANDS directory, or `.grb` sample. The `GRAPHCHALLENGE_REF_DIR` environment variable was also unset. No full multi-GB official archive was downloaded for this submission package; the official-data validation uses only the 256 MiB byte-range prefix described above.

## Interpretation for This Repo

- The deliverable is a paper-style GraphChallenge submission, not a leaderboard entry.
- The submission deadline for 2026 is July 7, 2026.
- The paper limit is six pages in IEEE HPEC style.
- The official examples include GraphBLAS-oriented implementations. This project deliberately keeps the primary path CPU-first and Python-first, using SciPy sparse matrices for accessibility.
- Large CAIDA/ILANDS PCAP data is not assumed to be fully present. The benchmark uses reproducible synthetic edge records plus a 5,000,000-packet byte-range prefix of the official capture, and leaves an optional reference wrapper in `scripts/run_reference.py`.
- The innovation claim is shape- and time-aware commodity CPU reproducibility: exact sparse analytics, multi-regime sparse traffic-matrix benchmarks, streaming/sketch heavy-hitter analytics, candidate-discovery sensitivity, early-stream method selection, time-bin network-sensing anomaly summaries, and interpretable sensing outputs.

## Conservative Claim Boundary

The current results are synthetic commodity-CPU benchmarks up to 5,000,000 edge records, streaming/sketch benchmarks up to 500,000 edge records, sketch-width and candidate-capacity sensitivity studies at 100,000 edge records, an early-stream method selector checked against saved 5,000,000-edge exact-method winners, controlled time-bin experiments for volume burst, scanner fanout, concentrated-destination, and distributed low-rate scan anomalies, a small Matrix Market official-format smoke test, and a 5,000,000-packet byte-range prefix of the official anonymized capture used to validate the parser, selector, certificate, and candidate-capacity findings on real challenge data. They can support claims about reproducibility, accessibility, multi-regime sparse matrix behavior, transparent sparse analytics, shape-dependent streaming approximation behavior, online method-choice diagnostics, sparse-coordinate format compatibility, interpretable temporal sensing summaries, and official-prefix validation of the selector and certificate. They do not support claims about Champion-level performance, full-capture (2^30-packet) throughput, or full CAIDA-scale processing.

Benchmark timing notes:

- Construction benchmarks time an already generated, in-memory edge table. They exclude synthetic data generation and disk I/O.
- CSV outputs now include total seconds, construction seconds, analytics seconds, edges per second, nnz per edge, and coarse process RSS per edge.
- The official-prefix fetch/parse time is measured and recorded in `data/real/official_prefix_manifest.json` (256 MiB Range download plus streaming decompression and packet parsing). This is a prefix parse-time observation, not a full-capture PCAP throughput claim.
