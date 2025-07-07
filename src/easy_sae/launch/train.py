import datasets
import torch
import transformers
from peft import get_peft_model

from easy_sae import TopKSaeConfig
from easy_sae.launch.config import ModelArguments, SaeConfig, TrainingArguments
from easy_sae.trainer import SaeTrainer
from easy_sae.utils.datasets import CacheDataset


def sae_mapping(sae_type, sae_config):
    if sae_type == "TOPK_SAE":
        sae_config = TopKSaeConfig(
            num_latents=sae_config.num_latents,
            k=sae_config.k,
            target_modules=sae_config.target_modules,
            task_type=sae_config.task_type,
        )
        return sae_config
    else:
        raise NotImplementedError(f"Sae type : {sae_type} not implemented ye")


def get_model(model_args, trainer_args):
    """
    A hard coded way to get model
    """
    model_name = model_args.model_path

    if "qwen2.5-vl" in model_name.lower():
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if trainer_args.bf16 else None),
        )
        processor = transformers.AutoProcessor.from_pretrained(model_name)
        tokenizer = transformers.AutoProcessor.from_pretrained(model_name)
    else:
        model = transformers.AutoModel.from_pretrained(
            model_name,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if trainer_args.bf16 else None),
        )
        processor = transformers.AutoProcessor.from_pretrained(model_name)
        tokenizer = transformers.AutoProcessor.from_pretrained(model_name)
    return model, processor, tokenizer


def main():
    parser = transformers.HfArgumentParser(
        [TrainingArguments, SaeConfig, ModelArguments]
    )
    trainer_args, sae_config, model_args = parser.parse_args_into_dataclasses()
    sae_type = sae_config.sae_type
    task_type = sae_config.task_type
    if task_type == "NONE":
        sae_config.task_type = None

    sae_config = sae_mapping(sae_type, sae_config)
    model, processor, tokenizer = get_model(model_args, trainer_args)
    model_config = model.config
    model.enable_input_require_grads()
    model = get_peft_model(model, sae_config)
    model.config = model_config
    model.print_trainable_parameters()

    dataset = trainer_args.dataset_path
    if "parquet" not in dataset:
        dataset = datasets.load_dataset(dataset)

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
