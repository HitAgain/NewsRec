#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

shell_dir=$(cd "$(dirname "$0")";pwd)
echo $shell_dir
cd $shell_dir
cd ../../miner
python3 miner.py \
    --root ../data \
    --split small \
    --mode train \
    --hist_max_len 20 \
    --backbone bert \
    --pretrain_model ../../Pretrain/bert-base-uncased/ \
    --glove_cache_dir ../data/glove/glove.6B/ \
    --output ./output
