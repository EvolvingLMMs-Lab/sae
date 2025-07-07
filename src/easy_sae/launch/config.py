from dataclasses import dataclass
from typing import Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_path: str = "./data/examples.parquet"
    text_key: Optional[str] = "text"
    image_key: Optional[str] = "images"
    video_key: Optional[str] = "videos"
    audio_key: Optional[str] = "audios"


@dataclass
class ModelArguments:
    model_path: str
    attn_implementation: str = "sdpa"


@dataclass
class SaeConfig:
    sae_type: str = "TOPK_SAE"
    num_latents: int = 4096
    k: Optional[int] = 32
    target_modules: Optional[str] = "model.layers.24.o_proj"
    task_type: str = "CAUSAL_LM"
