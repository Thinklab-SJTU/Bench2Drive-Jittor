#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/.../TCP
CUDA_VISIBLE_DEVICES=0 mpirun -np 1 python TCP/train.py