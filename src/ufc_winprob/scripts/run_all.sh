#!/usr/bin/env bash
set -euo pipefail

python -m ufc_winprob.pipelines.build_dataset
python -m ufc_winprob.models.training
python -m ufc_winprob.models.predict
python -m ufc_winprob.models.backtest
