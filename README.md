# Lightweight Anonymized Network Sensing Pipeline

This repository is a Student Innovation oriented entry for the MIT/IEEE/Amazon GraphChallenge Anonymized Network Sensing track. The contribution is shape- and time-aware commodity CPU reproducibility: exact sparse traffic-matrix analytics, multi-regime synthetic benchmarks, streaming/sketch heavy-hitter analytics, online method selection, time-bin anomaly summaries, and interpretable sensing outputs.

The project does not claim Champion-level throughput. It is designed to be easy to run, easy to inspect, and honest about commodity-hardware benchmark scale.

## What It Does

- Reads anonymized edge records with `src`, `dst`, and `bytes` columns.
- Builds sparse source-destination traffic matrices with `scipy.sparse`.
- Computes heavy hitters, source/destination degree and strength, density, entropy summaries, concentration, and a spectral statistic.
- Benchmarks four synthetic traffic regimes: `hotspot_zipf`, `community_bursty`, `scanner_fanout`, and `uniform_sparse`.
- Generates controlled synthetic samples with target `nnz`, density, duplication rate, skew, community structure, scanner/fanout behavior, and temporal bursts.
- Runs up to 5,000,000 input edges locally with different nnz/density profiles.
- Compares exact sparse analytics with a streaming Count-Min Sketch heavy-hitter path.
- Includes a sketch-width sensitivity study showing that larger sketches improve hotspot weight estimates but do not fix diffuse-regime heavy-hitter recall.
- Includes a candidate-capacity sensitivity study showing that diffuse-regime failure is largely candidate-discovery limited.
- Adds an early-stream method selector that recommends exact sparse vs pandas groupby using cheap traffic-shape features.
- Adds a conservative certified selector that reports prefix-based sketch safety conditions and writes v1 results under `results_v1/`.
- Fetches byte-range prefixes (256 MiB and 2 GiB) of the official Random PCAP data product (`pcap.zst`), parses 5,000,000-40,000,000 packet records in seconds, and validates the selector, safety screen, and candidate-capacity findings on organizer-published official data across eight disjoint stream segments.
- Cross-validates the pcap parser against the official GraphBLAS matrices: the first 2^23-packet segment matches exactly, 64/64 windows and 8,388,608 pairs (`scripts/grb_cross_validation.py`).
- Benchmarks a SuiteSparse:GraphBLAS construction baseline against the SciPy kernel, including native 2^32-index hypersparse construction (`scripts/benchmark_graphblas.py`).
- Computes per-time-bin sensing summaries for volume burst, scanner fanout, concentrated-destination, and distributed low-rate scan anomalies.
- Runs a small Matrix Market sparse-coordinate smoke test for GraphBLAS-style official-format compatibility.
- Compares our sparse path against transparent pandas/Python baselines and a GraphChallenge-paper-style reference formula compatibility check.
- Generates paper-ready CSV tables and figures from saved results.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make setup-check
make test
make smoke
make figures
```

If you prefer not to create a virtual environment, run the same `make` targets with a Python environment that already has NumPy, SciPy, pandas, and matplotlib installed.

## Main Commands

```bash
# Create a tiny reproducible synthetic edge file.
python3 scripts/download_sample_data.py --output data/sample/tiny_edges.csv --n-edges 5000

# Run the sparse analytics pipeline.
python3 scripts/run_ours.py --input data/sample/tiny_edges.csv --out-prefix results/tiny

# Run the multi-regime small benchmark suite.
python3 scripts/benchmark.py --output results/benchmark_summary.csv --sizes 5000 25000 100000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --repeats 3

# Generate figures from the saved benchmark CSV.
MPLCONFIGDIR=/tmp/graphsense-mpl python3 scripts/make_figures.py --input results/benchmark_summary.csv --outdir figures

# Run the large commodity-CPU benchmark.
make large-benchmark
MPLCONFIGDIR=/tmp/graphsense-mpl python3 scripts/make_figures.py --input results/benchmark_large_summary.csv --outdir figures

# Run the streaming/sketch benchmark.
make streaming-figures

# Run only the candidate-capacity sensitivity, method-selector, or time-bin anomaly experiment.
make candidate-sensitivity
make method-selector
make certified-selector
make certified-positive
make timebin-anomaly
make official-format-smoke

# Fetch a byte-range prefix of the official anonymized capture and validate on it.
# Downloads the first 256 MiB of the official pcap.zst (not the full 13.9 GB file),
# stream-decompresses the truncated frame, and parses packet records into
# data/real/official_prefix_edges.csv with a provenance manifest.
make fetch-official-prefix
make real-data

# Compare ours against the local GraphChallenge-paper-formula compatibility path.
python3 scripts/compare_reference.py --output results/reference_comparison.csv --n-edges 100000 --regime community_bursty
```

## Reproduce the Submission Figures

Run these commands from the repository root:

```bash
make setup-check
make test
make smoke
make figures
make large-figures
make streaming-figures
make method-selector
make certified-selector
make timebin-anomaly
make official-format-smoke
make reference
```

On the current commodity CPU run, `make large-benchmark` completed in roughly 10-15 seconds and the recorded process peak RSS in `results/benchmark_large_grouped.csv` stayed below about 1.8 GB. Benchmark runtimes exclude synthetic data generation and disk I/O; they time construction plus analytics on an in-memory edge table and also save construction-only time, analytics time, edges per second, nnz per edge, and coarse process RSS per edge. `make streaming-figures` can take several minutes because the streaming/sketch path and sensitivity scans use intentionally transparent Python loops, especially at large candidate capacities. Other machines and Python builds will differ, so treat these as rough expectations rather than fixed performance claims.

## Input Format

The primary input is a CSV or Parquet file with at least:

| column | meaning |
| --- | --- |
| `src` | anonymized source identifier |
| `dst` | anonymized destination identifier |
| `bytes` | nonnegative edge weight |

If `bytes` is absent but `packets` is present, the pipeline uses `packets`. If neither is present, each edge receives unit weight.

## Outputs

For an output prefix such as `results/tiny`, `scripts/run_ours.py` writes:

- `results/tiny_summary.csv`
- `results/tiny_top_edges.csv`
- `results/tiny_src_strength.csv`
- `results/tiny_dst_strength.csv`

The benchmark writes:

- `results/benchmark_summary.csv`
- `results/benchmark_grouped.csv`
- `results/benchmark_large_summary.csv`
- `results/benchmark_large_grouped.csv`
- `results/streaming_benchmark.csv`
- `results/streaming_sensitivity.csv`
- `results/candidate_sensitivity.csv`
- `results/method_selector_summary.csv`
- `results/real_data_summary.csv`
- `results/real_data_candidate_sensitivity.csv`
- `results_v1/certified_selector_summary.csv`
- `results_v1/certified_selector_tuned_summary.csv`
- `results_v1/certified_selector_official_summary.csv`
- `results/timebin_anomaly_summary.csv`
- `results/timebin_anomaly_scenarios.csv`
- `results/official_format_smoke.csv`
- `results/reference_comparison.csv`
- `figures/benchmark_runtime.png`
- `figures/benchmark_memory.png`
- `figures/benchmark_large_runtime.png`
- `figures/benchmark_large_memory.png`
- `figures/streaming_runtime.png`
- `figures/streaming_topk_recall.png`
- `figures/streaming_sensitivity.png`
- `figures/candidate_sensitivity.png`
- `figures/method_selector_features.png`
- `figures/timebin_anomaly.png`

## Official Reference and Data Notes

The official GraphChallenge data page lists Anonymized Network Sensing data products: the Random PCAP and Random GraphBLAS archives in the public `graphchallenge` S3 bucket (organizer-generated synthetic traffic in official formats), and a CAIDA-telescope-derived variant that is access-controlled and not used here. The full S3 files are 8-14 GB, so this repo does not download them in full. Instead, `scripts/fetch_official_prefix.py` Range-downloads a byte-range prefix of the official `pcap.zst` (256 MiB default; 2 GiB used for the segment-stability check), stream-decompresses the truncated zstd frame, parses the fixed-size packet records, and emits an anonymized `src,dst,bytes` edge table plus a provenance manifest (`data/real/official_prefix_manifest.json`, committed). The parsed identifiers are the challenge-provided anonymized addresses used as opaque integer labels; no re-anonymization is applied. `scripts/grb_cross_validation.py` additionally verifies the parser against the official GraphBLAS matrices (exact 64/64-window match on the first 2^23-packet segment; note the official matrices index addresses in host byte order). These prefixes validate the parser, the early-stream method selector, the safety screen, and the candidate-capacity findings on official data. They are prefixes of a synthetic official product, not the full capture and not real telescope traffic, and no full-capture throughput claim is made. The paper also describes reference sparse-matrix quantities for traffic-matrix sensing. This repository includes:

- `scripts/run_reference.py`, a wrapper for a separately checked-out official reference implementation.
- `scripts/official_format_smoke.py`, a small Matrix Market sparse-coordinate round-trip test for GraphBLAS-style matrix compatibility. It does not use official data.
- `scripts/compare_reference.py`, a local GraphChallenge-paper-formula compatibility check that computes traffic matrix counts and source/destination similarity nnz on a small reproducible sample. This is not the full official GraphChallenge reference repository.

If you have the official reference checkout, set:

```bash
export GRAPHCHALLENGE_REF_DIR=/path/to/reference
python3 scripts/run_reference.py -- make
```

This wrapper intentionally does not clone or download large challenge assets by default. A local search did not find an official checkout or small official sample in this workspace. See `docs/challenge_notes.md` for the official challenge, data, submission, and paper links.

## Paper Draft

The draft is in `paper/main.tex`. It is written as a conservative six-page IEEE/HPEC-style submission skeleton around the measured local benchmark results. Rerun the benchmark before final submission if the target machine changes.
