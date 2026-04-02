# OmniEval/framework/evaluator.py
import os
import sys
import json
import logging
import datetime
import importlib.util
from typing import List, Dict, Any
from tqdm import tqdm
from moviepy import VideoFileClip

from framework.prompt_utils import build_multiple_choice_prompt, extract_choice_answer

def setup_worker_logger(rank: int, log_dir: str) -> logging.Logger:
    logger = logging.getLogger(f"worker_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    log_file = os.path.join(log_dir, f'worker_rank_{rank}.log')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    if rank == 0:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(sh)
        
    return logger

def check_video_has_audio(video_path: str, logger: logging.Logger) -> bool:
    try:
        clip = VideoFileClip(video_path)
        has_audio = clip.audio is not None
        clip.close()
        return has_audio
    except Exception as e:
        logger.warning(f"Failed to check audio for {video_path}: {e}")
        return False

def evaluate_worker_process(
    rank: int, 
    device_ids: List[int], 
    data_shard: List[Dict], 
    wrapper_file: str, 
    class_name: str, 
    model_path: str, 
    video_dir: str, 
    log_dir: str, 
    return_dict: Any
):
    """Execution loop for a single worker process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    logical_device_ids = list(range(len(device_ids)))
    
    logger = setup_worker_logger(rank, log_dir)
    logger.info(f"[Rank {rank}] Started. Physical GPUs: {device_ids} -> Logical GPUs: {logical_device_ids}. Samples: {len(data_shard)}")
    
    wrapper_dir = os.path.dirname(os.path.abspath(wrapper_file))
    if wrapper_dir not in sys.path:
        sys.path.insert(0, wrapper_dir)
        
    try:
        spec = importlib.util.spec_from_file_location("user_wrapper_module", wrapper_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModelClass = getattr(module, class_name)
    except Exception as e:
        logger.error(f"[Rank {rank}] Failed to load user wrapper: {e}")
        raise e

    logger.info(f"[Rank {rank}] Initializing model on logical devices {logical_device_ids}...")
    try:
        model_instance = ModelClass(model_path=model_path, device_ids=logical_device_ids)
        logger.info(f"[Rank {rank}] Model initialized successfully.")
    except Exception as e:
        logger.error(f"[Rank {rank}] Model initialization failed: {e}")
        raise e

    results = []
    
    for i, sample in enumerate(tqdm(data_shard, desc=f"Rank {rank} Prog", position=rank)):
        video_id = sample.get("video_id", f"unknown_{i}")
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        result_entry = sample.copy()
        result_entry["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_entry["eval_mode"] = "standard"
        
        if not os.path.exists(video_path):
            logger.warning(f"[Rank {rank}] Video not found: {video_path}")
            result_entry.update({
                "model_prediction": None,
                "model_answer_raw": "SKIPPED: Video not found",
                "is_correct": False,
                "skipped": True,
                "skip_reason": "video_not_found"
            })
            results.append(result_entry)
            continue
            
        has_audio = check_video_has_audio(video_path, logger)
        
        prompt = build_multiple_choice_prompt(sample["question"], sample["options"])
        correct_option = sample.get("correct_option", "").strip().upper()
        
        try:
            raw_output = model_instance.generate(video_path=video_path, prompt=prompt, has_audio=has_audio)
            predicted_option = extract_choice_answer(raw_output)
            is_correct = (predicted_option == correct_option)
            
            result_entry.update({
                "model_prediction": predicted_option,
                "model_answer_raw": raw_output,
                "is_correct": is_correct,
                "skipped": False,
                "has_audio_track": has_audio
            })
            
            status = "✓" if is_correct else "✗"
            logger.info(f"[Rank {rank}] Video {video_id}: GT({correct_option}) -> Pred({predicted_option}) {status}")
            
        except Exception as e:
            logger.error(f"[Rank {rank}] Inference error on video {video_id}: {e}")
            result_entry.update({
                "model_prediction": None,
                "model_answer_raw": f"ERROR: {str(e)}",
                "is_correct": False,
                "skipped": True,
                "skip_reason": f"inference_error: {str(e)}"
            })
            
        results.append(result_entry)
        
        if (i + 1) % 10 == 0:
            intermediate_file = os.path.join(log_dir, f"intermediate_rank{rank}_{i+1}.json")
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    final_rank_file = os.path.join(log_dir, f"results_rank{rank}_final.json")
    with open(final_rank_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    return_dict[rank] = results
    logger.info(f"[Rank {rank}] Completed.")