.PHONY: demo compare power test compliance

PY ?= python

demo:
	$(PY) -m uplift.cli run --model t_learner --budget 4000

compare:
	$(PY) -m uplift.cli compare-models --budget 4000

power:
	$(PY) -m uplift.cli power

test:
	$(PY) -m pytest

compliance:
	$(PY) scripts/compliance_check.py .
