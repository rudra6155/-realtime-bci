# make install | test | benchmark | dashboard | profile | serve
PYTHON ?= python
# make test fails (instead of skipping) GPU tests when CUDA is missing. CI sets REQUIRE_CUDA=0.
REQUIRE_CUDA ?= 1

.PHONY: install test benchmark dashboard profile serve gif clean

install:
	$(PYTHON) -m pip install -e ".[dev]"
	-$(PYTHON) -m playwright install chromium || echo "WARN: browser download failed; test_dashboard.py needs: playwright install chromium"

test:
	BCI_REQUIRE_CUDA=$(REQUIRE_CUDA) $(PYTHON) -m pytest

benchmark:
	PYTHON=$(PYTHON) bash scripts/run_benchmark.sh

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
