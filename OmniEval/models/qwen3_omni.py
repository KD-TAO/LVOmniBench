# OmniEval/models/qwen3_omni.py
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
from framework.base_model import BaseOmniModel
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

class Qwen3OmniWrapper(BaseOmniModel):
    """
    Official Wrapper for Qwen3-Omni-30B (MoE) model.
    Supports multi-GPU loading for large parameters (e.g., 30B on 2 GPUs).
    """
    def load_model(self) -> None:
        device_map = "auto" if len(self.device_ids) > 1 else f"cuda:{self.device_ids[0]}"

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=device_map,
            attn_implementation="flash_attention_2"
        )
        self.logger.info(f"Qwen3-Omni loaded successfully.")

    def generate(self, video_path: str, prompt: str, has_audio: bool) -> str:
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError("Please ensure 'qwen_omni_utils.py' is in your PYTHONPATH or working directory.")

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please analyze the video carefully and select the most appropriate answer from the given options."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=has_audio)
        
        num_input_frames = videos[0].shape[0] if videos else 0
        if hasattr(self.model, 'thinker'):
            self.model.thinker.nframes = num_input_frames
            
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=has_audio
        ).to(self.primary_device)
        
        inputs = inputs.to(self.model.dtype)
        
        if not has_audio:
            self.logger.info("Audio input not detected; proceeding with video and text only.")

        text_ids, audio_out = self.model.generate(
            **inputs,
            speaker="Ethan",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=has_audio,
            return_audio=False,
            max_new_tokens=100,
            temperature=0.0,
        )
        
        generated_text = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return generated_text[0].strip() if generated_text else ""