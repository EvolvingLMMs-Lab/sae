import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    processor: Optional[ProcessorMixin] = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor is not None:
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        input_ids = inputs.pop("input_ids")
        input_ids = [input_id.squeeze(0) for input_id in input_ids]
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)
        inputs.pop("attention_mask")
        batched_inputs = {}
        for key, values in inputs.items():
            batched_inputs[key] = torch.concatenate(values, dim=0)
        batched_inputs["input_ids"] = input_ids
        batched_inputs["attention_mask"] = attention_mask

        return batched_inputs


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Union[HFDataset, str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_key: str,
        image_key: Optional[str] = None,
        video_key: Optional[str] = None,
        audio_key: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(dataset, str):
            dataset = HFDataset.from_parquet(dataset)

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
        self.text_key = text_key
        self.dataframe = dataset

    def __getitem__(self, index):
        row = self.dataframe[index]

        if self.processor is not None:
            # By default we assume
            text = self.processor.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            multi_modal_inputs = {}
            images = None
            if self.image_key in row:
                images = [image for image in row[self.image_key]]
                multi_modal_inputs["images"] = images

            # TODO
            # Implement the load logic for video and audios later
            if self.video_key in row:
                videos = [video for video in row[self.video_key]]
                multi_modal_inputs["videos"] = videos

            if self.audio_key in row:
                audios = [audio for audio in row[self.audio_key]]
                multi_modal_inputs["audios"] = audios

            model_inputs = self.processor(
                text=[text], return_tensors="pt", **multi_modal_inputs
            )
        else:
            text = self.tokenizer.apply_chat_template(
                row[self.text_key], tokenize=False, add_generation_prompt=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt")

        return model_inputs

    def get_collator(self):
        return DataCollator(self.tokenizer, self.processor)

    def __len__(self):
        return len(self.dataframe)
