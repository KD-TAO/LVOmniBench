# OmniEval/run_eval.py
import os
import sys
import json
import argparse
import datetime
import logging
import torch.multiprocessing as mp

# Ensure the framework can be imported regardless of where the script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from framework.evaluator import evaluate_worker_process
from framework.metrics import calculate_accuracy_metrics

def main():
    parser = argparse.ArgumentParser(description="OmniEval: Universal Audio-Video Multimodal Evaluation Framework")
    
    # Model configs
    parser.add_argument('--wrapper-file', type=str, required=True, help="Path to your model wrapper python file (e.g., models/qwen2_5_omni.py)")
    parser.add_argument('--class-name', type=str, required=True, help="Name of the wrapper class (e.g., Qwen2_5OmniWrapper)")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the model weights")
    
    # Data configs
    parser.add_argument('--data-path', type=str, required=True, help="Path to the flattened benchmark JSON file")
    parser.add_argument('--video-dir', type=str, required=True, help="Directory containing the video files")
    
    # Execution & Hardware configs
    parser.add_argument('--num-gpus', type=int, default=1, help="Total number of GPUs available for evaluation")
    parser.add_argument('--num-processes', type=int, default=1, help="Number of parallel worker processes. (e.g., 8 GPUs with 4 processes means 2 GPUs per model instance)")
    parser.add_argument('--mini-test-num', type=int, default=None, help="If set, only evaluate the first N samples for quick testing")
    
    # Output configs
    parser.add_argument('--output-dir', type=str, default="./results", help="Base directory to save final results")
    parser.add_argument('--run-name', type=str, default=None, help="Custom name for this evaluation run. Defaults to timestamp.")
    
    args = parser.parse_args()

    run_name = args.run_name if args.run_name else f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_output_dir = os.path.abspath(args.output_dir)
    run_output_dir = os.path.join(base_output_dir, run_name)
    logs_dir = os.path.join(run_output_dir, "logs")
    
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'main_orchestrator.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("OmniEval")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting OmniEval Framework")
    logger.info("=" * 60)
    logger.info(f"Wrapper File : {args.wrapper_file} ({args.class_name})")
    logger.info(f"Model Path   : {args.model_path}")
    logger.info(f"Data Path    : {args.data_path}")
    logger.info(f"Hardware     : {args.num_gpus} GPUs allocated across {args.num_processes} processes")
    logger.info(f"Output Dir   : {run_output_dir}")
    logger.info("=" * 60)

    if args.num_gpus < args.num_processes:
        raise ValueError(f"Number of GPUs ({args.num_gpus}) cannot be less than number of processes ({args.num_processes}).")
    
    gpus_per_process = args.num_gpus // args.num_processes
    process_device_map = {}
    
    for i in range(args.num_processes):
        start_id = i * gpus_per_process
        end_id = start_id + gpus_per_process
        process_device_map[i] = list(range(start_id, end_id))
        logger.info(f"Process [Rank {i}] assigned GPU IDs: {process_device_map[i]}")

    if args.num_gpus % args.num_processes != 0:
        logger.warning(f"Note: {args.num_gpus} GPUs are not perfectly divisible by {args.num_processes} processes. Some GPUs may be idle.")

    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
        
    if args.mini_test_num:
        benchmark_data = benchmark_data[:args.mini_test_num]
        logger.info(f"Mini test ON: Truncated to {args.mini_test_num} samples.")
        
    total_samples = len(benchmark_data)
    if total_samples == 0:
        logger.error("No samples found in the benchmark dataset.")
        sys.exit(1)

    samples_per_proc = (total_samples + args.num_processes - 1) // args.num_processes
    data_shards = []
    
    for i in range(args.num_processes):
        start_idx = i * samples_per_proc
        end_idx = min((i + 1) * samples_per_proc, total_samples)
        data_shards.append(benchmark_data[start_idx:end_idx])
        
    logger.info(f"Loaded {total_samples} samples. Sharded lengths: {[len(s) for s in data_shards]}")

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(args.num_processes):
        if len(data_shards[rank]) == 0:
            logger.info(f"Skipping Rank {rank} (no data assigned).")
            continue
            
        p = mp.Process(
            target=evaluate_worker_process,
            args=(
                rank, 
                process_device_map[rank], 
                data_shards[rank], 
                args.wrapper_file, 
                args.class_name, 
                args.model_path, 
                args.video_dir, 
                logs_dir, 
                return_dict
            )
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    logger.info("All worker processes have completed.")

    all_results = []
    for rank in range(args.num_processes):
        if rank in return_dict:
            all_results.extend(return_dict[rank])
            
    if not all_results:
        logger.error("No results were generated. Check worker logs for errors.")
        sys.exit(1)

    final_json_path = os.path.join(run_output_dir, "final_predictions.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    metrics = calculate_accuracy_metrics(all_results)
    metrics_path = os.path.join(logs_dir, "accuracy_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    logger.info("=" * 60)
    logger.info("🎉 EVALUATION COMPLETE 🎉")
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['total_correct']}/{metrics['total_samples']})")
    logger.info(f"Total Skipped/Errors: {metrics['total_skipped']}")
    logger.info(f"Full Predictions saved to: {final_json_path}")
    logger.info(f"Detailed Metrics saved to: {metrics_path}")
    logger.info(f"Logs saved in: {logs_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()