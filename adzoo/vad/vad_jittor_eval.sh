#!/usr/bin/env bash

T=`date +%m%d%H%M`c
# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
if [ "$GPUS" -lt 8 ]; then
  GPUS_PER_NODE=$GPUS
else
  GPUS_PER_NODE=8
fi


WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to VAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

export JT_SAVE_MEM=1
# export lazy_execution=0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
echo ${GPUS_PER_NODE}
mpirun -np ${GPUS_PER_NODE}  python   \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T
