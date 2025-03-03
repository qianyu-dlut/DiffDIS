#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python ../run_train.py \
--batchsize 4 \
--trainsize 1024 \
--decay_rate 0.95 \
--decay_epoch 30 \
--pretrained_model_name_or_path "/path/to/sd-turbo/" \
--dataset_path "/path/to/DIS5K/" \