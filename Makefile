PYTHON ?= python3

.PHONY: audit-vnext audit-vnext-with-paper

audit-vnext:
	SOW_ALLOW_INFERENCE=0 $(PYTHON) scripts/audit/reproduce_vnext.py

audit-vnext-with-paper:
	SOW_ALLOW_INFERENCE=0 $(PYTHON) scripts/audit/reproduce_vnext.py --build-paper

