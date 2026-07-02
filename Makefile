PYTHON ?= python3
MPLCONFIGDIR ?= /tmp/graphsense-mpl
PYTHONPYCACHEPREFIX ?= /tmp/graphsense-pycache

.PHONY: setup-check sample benchmark streaming-benchmark streaming-sensitivity candidate-sensitivity timebin-anomaly method-selector certified-selector certified-positive official-format-smoke large-benchmark reference fetch-official-prefix real-data window-stability conformal-certificate dyadic-comparison figures large-figures streaming-figures paper-numbers smoke test clean

setup-check:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/setup_check.py

sample:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/download_sample_data.py --output data/sample/tiny_edges.csv --n-edges 5000

benchmark: sample
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark.py --output results/benchmark_summary.csv --sizes 5000 25000 100000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --repeats 3

large-benchmark:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark.py --output results/benchmark_large_summary.csv --sizes 1000000 5000000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --methods sparse_direct pandas_groupby --repeats 3

streaming-benchmark:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark_streaming.py --output results/streaming_benchmark.csv --sizes 25000 100000 500000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse

streaming-sensitivity:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark_streaming_sensitivity.py --output results/streaming_sensitivity.csv --n-edges 100000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --widths 512 2048 8192 32768
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_streaming_sensitivity_figure.py --input results/streaming_sensitivity.csv --outdir figures

candidate-sensitivity:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark_candidate_sensitivity.py --output results/candidate_sensitivity.csv --n-edges 100000 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --capacities 64 128 512 2048 8192
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_candidate_sensitivity_figure.py --input results/candidate_sensitivity.csv --outdir figures

timebin-anomaly:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/timebin_anomaly.py --output results/timebin_anomaly_summary.csv --figure figures/timebin_anomaly.png

method-selector:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/method_selector.py --output results/method_selector_summary.csv --benchmark-csv results/benchmark_large_summary.csv --n-edges 5000000 --prefix-edges 50000 --materialize-full-stream
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_method_selector_figure.py --input results/method_selector_summary.csv --outdir figures

certified-selector:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/certified_selector.py --output results_v1/certified_selector_summary.csv --n-edges 5000000 --prefix-edges 50000 --top-k 20 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse

certified-positive:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/certified_selector.py --output results_v1/certified_selector_tuned_summary.csv --n-edges 5000000 --prefix-edges 50000 --top-k 20 --regimes hotspot_zipf community_bursty scanner_fanout uniform_sparse --current-width 16384 --current-candidate-capacity 5000
	@test -f data/real/official_prefix_edges.csv && PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/certified_selector.py --inputs data/real/official_prefix_edges.csv --output results_v1/certified_selector_official_summary.csv --prefix-edges 50000 --top-k 20 || echo "data/real/official_prefix_edges.csv missing; skipping official-data certificate (run make fetch-official-prefix)"

fetch-official-prefix:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/fetch_official_prefix.py --bytes 268435456 --max-packets 5000000 --output data/real/official_prefix_edges.csv

real-data:
	@test -f data/real/official_prefix_edges.csv || (echo "data/real/official_prefix_edges.csv missing; run make fetch-official-prefix first" && exit 1)
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/real_data_experiment.py --input data/real/official_prefix_edges.csv --summary-output results/real_data_summary.csv --sensitivity-output results/real_data_candidate_sensitivity.csv

window-stability:
	@test -f data/real/official_prefix40m_edges.csv || (echo "data/real/official_prefix40m_edges.csv missing; run scripts/fetch_official_prefix.py --bytes 2147483648 --max-packets 40000000 --output data/real/official_prefix40m_edges.csv --manifest data/real/official_prefix40m_manifest.json first" && exit 1)
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/window_stability.py

conformal-certificate:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/conformal_certificate.py --inputs data/real/ctu13_edges.csv data/real/official_prefix_edges.csv --budgets 8192:5:512 16384:5:5000 262144:5:98304 524288:5:200000

dyadic-comparison:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/benchmark_dyadic.py --inputs data/real/ctu13_edges.csv data/real/official_prefix_edges.csv

official-format-smoke:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/official_format_smoke.py --matrix-output data/official_format/tiny_traffic_matrix.mtx --summary-output results/official_format_smoke.csv

reference:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/compare_reference.py --output results/reference_comparison.csv --n-edges 100000 --regime community_bursty

figures: benchmark reference
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_figures.py --input results/benchmark_summary.csv --outdir figures

large-figures: large-benchmark
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_figures.py --input results/benchmark_large_summary.csv --outdir figures

streaming-figures: streaming-benchmark streaming-sensitivity candidate-sensitivity
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/make_streaming_figures.py --input results/streaming_benchmark.csv --outdir figures

paper-numbers: figures large-figures streaming-figures timebin-anomaly method-selector certified-selector certified-positive official-format-smoke
	@echo "all paper-facing CSVs and figures regenerated"

smoke: setup-check sample
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) scripts/run_ours.py --input data/sample/tiny_edges.csv --out-prefix results/tiny

test:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m unittest discover -s tests

clean:
	rm -f data/sample/*.csv data/synthetic/*.csv results/*.csv results/*.json figures/*.png figures/*.pdf
