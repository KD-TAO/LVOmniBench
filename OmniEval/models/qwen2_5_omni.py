# OmniEval/models/qwen2_5_omni.py
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from framework.base_model import BaseOmniModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

class Qwen2_5OmniWrapper(BaseOmniModel):
    """
    Official Wrapper for Qwen2.5-Omni model.
    """
    def load_model(self) -> None:
        device_map = f"cuda:{self.device_ids[0]}" if self.device_ids else "cpu"
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2"
        )
        self.logger.info(f"Qwen2.5-Omni loaded successfully on {device_map}.")

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
        
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=has_audio
        ).to(self.model.device).to(self.model.dtype)
        
        if not has_audio:
            self.logger.info("Audio input not detected; proceeding with video and text only.")
    
        text_ids = self.model.generate(
            **inputs,
            use_audio_in_video=has_audio,
            return_audio=False,
            max_new_tokens=100,
            temperature=0.0,
        )
        
        generated_text = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return generated_text[0].strip() if generated_text else ""