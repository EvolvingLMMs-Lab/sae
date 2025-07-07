from dataclasses import dataclass
from typing import Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_path: str
    text_key: Optional[bool] = "text"
    image_key: Optional[bool] = "images"
    video_key: Optional[bool] = "videos"
    audio_key: Optional[bool] = "audios"


@dataclass
class ModelArguments:
    model_path: str
    attn_implementation: str


@dataclass
class SaeConfig:
    sae_type: str = "TOPK_SAE"
    num_latents: int = 4096
    k: Optional[int] = 32
