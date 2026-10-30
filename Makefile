# make install | test | benchmark | benchmark-real | dashboard | profile | serve
PYTHON ?= python
# make test fails (instead of skipping) GPU tests when CUDA is missing. CI sets REQUIRE_CUDA=0.
REQUIRE_CUDA ?= 1

.PHONY: install test test-real benchmark benchmark-real dashboard profile serve gif clean

install:
	$(PYTHON) -m pip install -e ".[dev]"
	-$(PYTHON) -m playwright install chromium || echo "WARN: browser download failed; test_dashboard.py needs: playwright install chromium"

test:
	BCI_REQUIRE_CUDA=$(REQUIRE_CUDA) $(PYTHON) -m pytest

benchmark:
	PYTHON=$(PYTHON) bash scripts/run_benchmark.sh

# Real EEG (BCI Competition IV 2a, downloads ~700 MB via MOABB on first use): accuracy + kappa for 9 subjects,
# then label latency and LSL jitter on the GPU -> results/real/
benchmark-real:
	$(PYTHON) -m bci.real_benchmark all --out-dir results/real

# also run the tests that need the real dataset
test-real:
	BCI_RUN_REAL=1 BCI_REQUIRE_CUDA=$(REQUIRE_CUDA) $(PYTHON) -m pytest

dashboard:
	$(PYTHON) -m streamlit run app/dashboard.py

profile:
	$(PYTHON) scripts/profile.py

serve:
	$(PYTHON) -m bci.server --host 127.0.0.1 --port 8000

gif:
	$(PYTHON) scripts/record_gif.py --out results/dashboard_demo.gif

clean:
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
