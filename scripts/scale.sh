#!/usr/bin/env bash

set -ex


LR=0.9
LR_MAX_STEPS=10
LR_PATIENCE=500
LR_FACTOR=0.9
EPOCHS=15000
PPD=1

RUN_EXP="python -m experiments.main"
COMMON_ARGS="--epochs=$EPOCHS --lr=$LR --lr-max-steps=$LR_MAX_STEPS --lr-patience=$LR_PATIENCE --lr-factor=$LR_FACTOR"

# N
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="N" -N 4000 -N 8000 -N 16000 -N 32000 -N 64000 -N 128000 -T 50 -d 2 -k 10

# d
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="d" -N 10000 -T 10 -d 1 -d 2 -d 3 -d 4 -d 5 -k 10

# k
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="k" -N 10000 -T 50 -d 2 -k 1 -k 2 -k 4 -k 8 -k 16 -k 32 -k 64 -k 128 -k 256 -k 512 -k 1024

# T
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="T" -N 10000 -T 10 -T 20 -T 30 -T 40 -T 50 -T 60 -T 70 -T 80 -T 90 -T 100 -k 2 -d 2

# SGD on N
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="SGD-N" -N 50000 -N 100000 -N 1000000 -N 2000000 -T 50 -d 2 -k 10 --bs-y 50000

# SGD on U
$RUN_EXP -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="SGD-U" -N 100000 -T 50 -T 60 -T 70 -T 80 -T 90 -T 100 -d 2 -k 10 --bs-u 2500
