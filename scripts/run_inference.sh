#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python ../run_inference.py \
--denoise_steps 1 \
--ensemble_size 1 \
--processing_res 1024 \
--checkpoint_path "/path/to/your/checkpoint/" \
--pretrained_model_path "/path/of/sd-turbo/" \
--output_dir "/path/to/save/outputs/"
