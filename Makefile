PY ?= .venv/bin/python

data:        ## simulate all splits (configs/data.yaml) -> data/v1 with manifest
	$(PY) scripts/make_data.py
global:      ## LightGBM / XGBoost / MLP global models
	$(PY) scripts/train_global.py lgb,xgb,mlp all
attention:   ## meta-train the site-adaptive encoder-decoder attention model
	$(PY) scripts/train_attention.py 12000
evaluate:    ## methods x calibration budget x protocol on unseen + stress environments
	$(PY) scripts/eval_sites.py
studies:     ## split leakage, distribution shift, sensor ablations
	$(PY) scripts/split_study.py && cd scripts && ../$(PY) shift_study.py
	for fs in no_imu no_ref lag0; do $(PY) scripts/train_global.py lgb $$fs; done
package:     ## ONNX export + manifest + validation (fails on any broken invariant)
	$(PY) scripts/package_model.py
report:
	$(PY) scripts/make_report.py
check:
	$(PY) -m ruff check emtrack scripts tests && $(PY) -m mypy && $(PY) -m pytest -q
all: data global attention evaluate studies report package check

.PHONY: data global attention evaluate studies package report check all
