#!/bin/bash
# =============================================================================
# OmniEval: Sanity check script (single-video smoke test)
# Please modify the paths below to match your environment before running.
# =============================================================================

python sanity_check.py \
    --wrapper-file models/qwen2_5_omni.py \
    --class-name Qwen2_5OmniWrapper \
    --model-path /path/to/Qwen2.5-Omni-3B \
    --video-path /path/to/test_video.mp4 \
    --device-ids 0 \
    --prompt "Provide audio transcription text."
