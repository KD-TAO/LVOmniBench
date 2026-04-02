#!/bin/bash
# =============================================================================
# OmniEval: Example evaluation script
# Please modify the paths below to match your environment before running.
# =============================================================================

# --- Qwen2.5-Omni (single GPU per process) ---
# python run_eval.py \
#     --wrapper-file models/qwen2_5_omni.py \
#     --class-name Qwen2_5OmniWrapper \
#     --model-path /path/to/Qwen2.5-Omni-3B \
#     --data-path /path/to/data.json \
#     --video-dir /path/to/videos \
#     --num-gpus 8 \
#     --num-processes 8 \
#     --output-dir ./results \
#     --run-name qwen2_5_omni_eval

# --- Qwen3-Omni-30B MoE (2 GPUs per process) ---
python run_eval.py \
    --wrapper-file models/qwen3_omni.py \
    --class-name Qwen3OmniWrapper \
    --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --data-path /path/to/data.json \
    --video-dir /path/to/videos \
    --num-gpus 8 \
    --num-processes 4 \
    --output-dir ./results \
    --run-name qwen3_omni_eval
