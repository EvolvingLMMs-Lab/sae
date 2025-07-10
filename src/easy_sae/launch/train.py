import datasets
import torch
import transformers
import wandb
import os

from easy_sae import get_peft_sae_model
from easy_sae.launch.config import ModelArguments, SaeConfig, TrainingArguments
from easy_sae.trainer import SaeTrainer
from easy_sae.utils import hf_processor, hf_tokenizer
from easy_sae.utils.datasets import CacheDataset
from easy_sae.utils.factory import ModelFactory, SaeFactory

try:
    if not os.environ.get("WANDB_API_KEY", None):
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
except Exception as e:
    pass

def main():
    parser = transformers.HfArgumentParser(
        [TrainingArguments, SaeConfig, ModelArguments]
    )
    trainer_args, sae_config, model_args = parser.parse_args_into_dataclasses()
    sae_type = sae_config.sae_type

    sae_config = SaeFactory.create_sae_config(
        sae_type=sae_type,
        sae_config=sae_config.__dict__,
    )

    processor = hf_processor(model_args.model_path)
    tokenizer = hf_tokenizer(model_args.model_path)

    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch.bfloat16 if trainer_args.bf16 else torch.float32,
        "model_name": model_args.model_path,
    }

    model = ModelFactory.create_model(**model_kwargs)
    model_config = model.config
    model = get_peft_sae_model(model, sae_config)
    model.config = model_config
    model.print_trainable_parameters()

    dataset = trainer_args.dataset_path
    split = trainer_args.split
    subset = trainer_args.subset
    if "parquet" not in dataset: # if dataset is not processed, load it from huggingface and process it in local
        dataset = datasets.load_dataset(dataset, split=split, name=subset)
        if "LLaVA-OneVision-Data" in dataset:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "examples"))
            from data_process.llava_ov_clevr import convert_llava_ov_dataset
            dataset = dataset.map(convert_llava_ov_dataset, remove_columns=dataset.column_names, num_proc=32)

    sae_dataset = CacheDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
        text_key=trainer_args.text_key,
        image_key=trainer_args.image_key,
        video_key=trainer_args.video_key,
        audio_key=trainer_args.audio_key,
    )

    trainer = SaeTrainer(
        model=model,
        args=trainer_args,
        data_collator=sae_dataset.get_collator(),
        train_dataset=sae_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
