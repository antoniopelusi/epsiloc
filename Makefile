.PHONY: all setup privatize stats metadata format
.SILENT: all setup privatize stats metadata format

all:
	echo "|> No target selected. Use: setup, privatize, stats, metadata, format"

VENV = .venv
ACTIVATE = . $(VENV)/bin/activate
CONFIG = iris
PY_FILES = epsiloc.py epsiloc

setup:
	echo "=========| setup started... |========="
	echo "  cleaning old virtual environment..."
	rm -rf $(VENV)
	echo "  creating virtual environment..."
	python3 -m venv $(VENV)
	echo "  installing requirements..."
	$(ACTIVATE) && python3 -m pip install -r requirements.txt
	echo "=========| setup complete |========="

privatize:
	$(ACTIVATE) && python3 epsiloc.py $(CONFIG) --privatize

stats:
	$(ACTIVATE) && python3 epsiloc.py $(CONFIG) --stats

metadata:
	$(ACTIVATE) && python3 epsiloc.py $(CONFIG) --metadata

format:
	echo "=========| format started... |========="
	echo "  sorting python imports..."
	$(ACTIVATE) && ruff check --select I --fix $(PY_FILES)
	echo "  formatting python code..."
	$(ACTIVATE) && ruff format $(PY_FILES)
	echo "=========| format complete |========="
