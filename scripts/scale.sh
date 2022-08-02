#!/usr/bin/env bash

set -ex


LR=0.9
LR_MAX_STEPS=10
LR_PATIENCE=500
LR_FACTOR=0.9
EPOCHS=20000
PPD=1
DEVICES=4,5,6

COMMON_ARGS="--epochs=$EPOCHS --lr=$LR --lr-max-steps=$LR_MAX_STEPS --lr-patience=$LR_PATIENCE --lr-factor=$LR_FACTOR"

# N
#python main.py -g --ppd $PPD --out-tag="N" scale-exp $COMMON_ARGS -N 4000 -N 8000 -N \
# 16000 -N 32000 -N 64000 -N 128000 -T 50 -d 2 -k 10

## d
#python main.py -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="d" -N 10000 -T 10 -d 1 -d 2 -d 3 -d 4 -d 5 -k 10
#
## k
#python main.py -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="k" -N 10000 -T 50 -d 2 -k 1 -k 2 -k 4 -k 8 -k 16 -k 32 -k 64 -k 128 -k 256 -k 512 -k 1024
#
## T
#python main.py -g --ppd $PPD --out-tag="T" scale-exp $COMMON_ARGS -N 10000 -T 10 \
#-T 20 -T 30 -T 40 -T 50 -T 60 -T 70 -T 80 -T 90 -T 100 -k 2 -d 2

# SGD on N
python main.py -g --devices $DEVICES --ppd $PPD --out-tag="SGDN" scale-exp \
$COMMON_ARGS \
-N 25000 -N 50000 -N 75000 -N 90000 -N 100000 -N 200000 -N 300000 -T 50 -d 2 -k 10 \
--bs-y 50000

python main.py -g --devices $DEVICES --ppd $PPD --out-tag="SGDN" scale-exp \
$COMMON_ARGS -N 500000 -N 750000 \
-N 1000000 -N 1250000 -N 1500000 -N 2000000 -T 50 -d 2 -k 10 --bs-y 50000

# SGD on U
#python main.py -g --ppd $PPD scale-exp $COMMON_ARGS --out-tag="SGD-U" -N 100000 -T 50 -T 60 -T 70 -T 80 -T 90 -T 100 -d 2 -k 10 --bs-u 2500
