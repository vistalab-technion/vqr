#!/usr/bin/env bash

set -ex


LR=0.9
LR_MAX_STEPS=10
LR_PATIENCE=500
LR_FACTOR=0.9
EPOCHS=15000
PPD=1

RUN_EXP="python -m experiments.main"
COMMON_ARGS="--epochs=5000 --lr=$LR --lr-max-steps=$LR_MAX_STEPS \
--lr-patience=$LR_PATIENCE --lr-factor=$LR_FACTOR"
DP_COMMON_ARGS="--dp-epochs=$EPOCHS --dp-lr=$LR --dp-lr-max-steps=$LR_MAX_STEPS \
--dp-lr-patience=$LR_PATIENCE --dp-lr-factor=$LR_FACTOR"

# Epsilon experiment (d=1)
$RUN_EXP -g --ppd $PPD --out-tag="eps-1" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 100 -d 1 -k 1 -N 20000 --t-factor 100 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 0.99 -E 5e-1 -E 25e-2 -E 1e-1 -E 5e-2 -E 25e-3 -E 1e-2 -E 5e-3 -E 25e-4 -E 1e-3 \
 -E 5e-4 -E 25e-5 -E 1e-4

# Epsilon experiment (d=2)
$RUN_EXP -g --ppd $PPD --out-tag="eps-2" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 50 -d 2 -k 1 -N 20000 --t-factor 2 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 0.99 -E 5e-1 -E 25e-2 -E 1e-1 -E 5e-2 -E 25e-3 -E 1e-2 -E 5e-3 -E 25e-4 -E 1e-3 \
 -E 5e-4 -E 25e-5 -E 1e-4

# SGD-N experiment (d=1)
$RUN_EXP -g --ppd $PPD --out-tag="sgdn-1" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 100 -d 1 -k 1 -N 20000 --t-factor 100 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 1e-2 --bs-y 1000 --bs-y 2500 --bs-y 5000 --bs-y 10000 \
 --bs-y 20000

# SGD-N experiment (d=2)
$RUN_EXP -g --ppd $PPD --out-tag="sgdn-2" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 50 -d 2 -k 1 -N 20000 --t-factor 2 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 1e-2 --bs-y 1000 --bs-y 2500 --bs-y 5000 --bs-y 10000 \
 --bs-y 20000

# SGD-U experiment (d=1)
$RUN_EXP -g --ppd $PPD --out-tag="sgdu-1" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 100 -d 1 -k 1 -N 20000 --t-factor 100 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 1e-2 --bs-u 10 --bs-u 25 --bs-u 50 --bs-u 75 --bs-u 100

# SGD-U experiment (d=2)
$RUN_EXP -g --ppd $PPD --out-tag="sgdu-2" optim-exp $COMMON_ARGS $DP_COMMON_ARGS \
 -T 50 -d 2 -k 1 -N 20000 --t-factor 2 \
 -dp_N 100000 --n-eval-x 100 -dp_E 1e-3 \
 -E 1e-2 --bs-u 1000 --bs-u 1500 --bs-u 2000 --bs-u 2500
