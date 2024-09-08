#!/bin/bash

# infer for submitting "prediction.txt" to Leadboard
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
shell_dir=$(cd "$(dirname "$0")";pwd)
cd $shell_dir
cd ../../miner
python3 miner.py \
    --root ../data \
    --split large \
    --mode infer \
    --hist_max_len 20 \
    --backbone bert \
    --batch_size 8 \
    --pretrain_model ../../Pretrain/bert-base-uncased/ \
    --glove_cache_dir ../data/glove/glove.6B/ \
    --output ./output
