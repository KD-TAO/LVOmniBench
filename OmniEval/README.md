# OmniEval

Evaluation framework for [LVOmniBench](https://github.com/KD-TAO/LVOmniBench). Supports multi-GPU distributed inference with plug-and-play model adaption.

## Quickstart

### Step 0. Installation

```bash
git clone https://github.com/KD-TAO/LVOmniBench.git
cd LVOmniBench/OmniEval
# Use the official env settings.
```

OmniEval itself is lightweight. For **model-specific dependencies**, please follow the official installation guide of the model you want to evaluate. For example:
- **Qwen2.5-Omni**: Follow the [Qwen2.5-Omni] setup guide to install `qwen_omni_utils`, `flash-attn`, etc.
- **Qwen3-Omni**: Follow the [Qwen3-Omni] setup guide.

### Step 1. Download Data

Download the LVOmniBench benchmark from [HuggingFace](https://huggingface.co/datasets/KD-TAO/LVOmniBench). The dataset directory should look like:

```text
LVOmniBench/
├── data.json          # 1,014 multiple-choice QA samples
└── videos/            # 275 long-form videos (10–90 min each)
    ├── video_0.mp4
    ├── video_1.mp4
    └── ...
```

### Step 2. Sanity Check (Recommended)

Before running full evaluation, verify your model wrapper works with a single video:

```bash
python sanity_check.py \
    --wrapper-file models/qwen2_5_omni.py \
    --class-name Qwen2_5OmniWrapper \
    --model-path /path/to/Qwen2.5-Omni-3B \
    --video-path /path/to/test_video.mp4 \
    --device-ids 0 \
    --prompt "What is happening in this video?"
```

### Step 3. Full Evaluation

Use `run_eval.py` to run distributed evaluation across multiple GPUs.

**Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--wrapper-file` | Path to your model wrapper `.py` file | *required* |
| `--class-name` | Name of the wrapper class | *required* |
| `--model-path` | Path to the model weights | *required* |
| `--data-path` | Path to the benchmark JSON file | *required* |
| `--video-dir` | Directory containing the video files | *required* |
| `--num-gpus` | Total number of GPUs available | `1` |
| `--num-processes` | Number of parallel worker processes | `1` |
| `--output-dir` | Base directory to save results | `./results` |
| `--run-name` | Custom name for this run | timestamp |
| `--mini-test-num` | Only evaluate first N samples (for debugging) | `None` |

**Example: Qwen2.5-Omni on 8 GPUs (Data Parallelism)**

Each process loads one model instance on a single GPU:

```bash
python run_eval.py \
    --wrapper-file models/qwen2_5_omni.py \
    --class-name Qwen2_5OmniWrapper \
    --model-path /path/to/Qwen2.5-Omni-3B \
    --data-path /path/to/data.json \
    --video-dir /path/to/videos \
    --num-gpus 8 \
    --num-processes 8 \
    --output-dir ./results \
    --run-name qwen2_5_omni_eval
```

**Example: Qwen3-Omni-30B MoE on 8 GPUs (Tensor Parallelism)**

The 30B MoE model requires 2 GPUs per instance, so 8 GPUs give 4 parallel workers:

```bash
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
```

### GPU Allocation Strategy

OmniEval evenly distributes GPUs across worker processes and **physically isolates** them via `CUDA_VISIBLE_DEVICES`. This means:

- `8 GPUs / 8 processes` = 1 GPU per process (Data Parallelism)
- `8 GPUs / 4 processes` = 2 GPUs per process (Tensor Parallelism for large models)
- `8 GPUs / 2 processes` = 4 GPUs per process (for very large models)

Each worker process only sees its assigned GPUs as logical device `cuda:0`, `cuda:1`, ..., ensuring complete isolation and no cross-process OOM.

### Output Structure

After evaluation completes, results are saved under `--output-dir`:

```text
results/<run-name>/
├── final_predictions.json            # All predictions with correctness labels
└── logs/
    ├── accuracy_metrics.json         # Overall & fine-grained accuracy
    ├── main_orchestrator.log         # Main process log
    ├── worker_rank_0.log             # Per-worker logs
    ├── results_rank0_final.json      # Per-worker results
    └── intermediate_rank0_10.json    # Checkpoints (every 10 samples)
```

---

## Adapting Your Own Model

To evaluate a new model, create a wrapper by inheriting `BaseOmniModel` and implementing two methods:

### Step 1. Create a Wrapper File

Create a new file (e.g., `models/my_model.py`):

```python
import torch
from framework.base_model import BaseOmniModel

class MyModelWrapper(BaseOmniModel):

    def load_model(self) -> None:
        """
        Load model weights onto the assigned GPUs.

        Available attributes:
            self.model_path   — path to model weights
            self.device_ids   — list of logical GPU IDs (e.g. [0, 1])
            self.primary_device — "cuda:0" (first assigned GPU)
        """
        # For single-GPU models:
        self.model = YourModel.from_pretrained(
            self.model_path,
            device_map=self.primary_device,
            torch_dtype=torch.bfloat16
        )
        # For multi-GPU models, use device_map="auto":
        # self.model = YourModel.from_pretrained(
        #     self.model_path,
        #     device_map="auto"
        # )

    def generate(self, video_path: str, prompt: str, has_audio: bool) -> str:
        """
        Run inference on a single video sample.

        Args:
            video_path: Absolute path to the video file (.mp4)
            prompt:     Pre-formatted question + options text
            has_audio:  Whether the video contains an audio track

        Returns:
            Raw text output from the model (answer extraction is handled by the framework).
        """
        # Your inference logic here
        # Use `has_audio` to decide whether to process the audio modality
        output = self.model.inference(video_path, prompt)
        return output
```

### Step 2. Run Sanity Check

```bash
python sanity_check.py \
    --wrapper-file models/my_model.py \
    --class-name MyModelWrapper \
    --model-path /path/to/weights \
    --video-path /path/to/test_video.mp4 \
    --device-ids 0
```

### Step 3. Run Full Evaluation

```bash
python run_eval.py \
    --wrapper-file models/my_model.py \
    --class-name MyModelWrapper \
    --model-path /path/to/weights \
    --data-path /path/to/data.json \
    --video-dir /path/to/videos \
    --num-gpus 4 \
    --num-processes 2 \
    --run-name my_model_eval
```

### Notes on Wrapper Implementation

- **GPU isolation is automatic**: OmniEval sets `CUDA_VISIBLE_DEVICES` before your wrapper is loaded. Your `device_ids` are always logical IDs starting from `0`. Just use them directly.
- **Answer extraction is automatic**: Return the raw model output from `generate()`. The framework extracts A/B/C/D answers using a multi-pattern regex engine.
- **Audio detection is automatic**: The `has_audio` flag is pre-computed by the framework. Use it to conditionally enable your model's audio pipeline.
- **Dynamic path injection**: Your wrapper file can live anywhere — OmniEval automatically adds its directory to `sys.path` so your local imports work.

---

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{tao2026lvomnibench,
  title={LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs},
  author={Tao, Keda and Zheng, Yuhua and Xu, Jia and Du, Wenjie and Shao, Kele and Wang, Hesong and Chen, Xueyi and Jin, Xin and Zhu, Junhan and Yu, Bohan and others},
  journal={arXiv preprint arXiv:2603.19217},
  year={2026}
}
```
