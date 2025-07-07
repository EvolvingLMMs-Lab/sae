# Easy-SAE
Easy-SAE is inspired by a wealth of Sparse Autoencoder (SAE) work from Anthropic, OpenAI, Google, and the open-source community. SAE has become a powerful and widely-used tool in the field of explainable AI. This project aims to provide a simple and flexible interface that allows users to inject SAE modules into their models at any layer with minimal effort. As long as the target is an nn.Module, SAE can be easily integrated and trained.

## Design Philosophy
The code design takes inspiration from PEFT, as we believe SAE shares many structural similarities with PEFT-based methods. By inheriting from a BaseTuner class, we enable seamless SAE integration into existing models.

With this design, injecting an SAE module is as simple as:

```python

import torch
import torch.nn as nn
from peft import get_peft_model, inject_adapter_in_model

from easy_sae import TopKSaeConfig

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

# Inject the adapter into the model
model = inject_adapter_in_model(config, model)

# Check if the adapter was injected correctly
result = model(torch.randn(1, 512, 10))
```

You can also obtain a PEFT-wrapped model using the magic function from the PEFT library. The rest of your workflow remains the same:

```python
# Get the PEFT model
peft_model = get_peft_model(model, config)

# Allows you to cache the output and input state
kwargs = {"with_cache": True}
peft_model.special_peft_forward_args.add("with_cache")

result = peft_model(torch.randn(1, 512, 10), **kwargs)

self.assertIsInstance(result, torch.Tensor)
self.assertEqual(result.shape, (1, 512, 10))

```
## Training

We provide a simple training recipe to help you get started quickly. You’re also free to implement your own training pipeline.

```bash
torchrun --nproc_per_node="1" --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="1234" \
    src/easy_sae/launch/train.py \
    --dataset_path ./data/mmmu.parquet \
    --image_key images \
    --run_name sae_test \
    --report_to none \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --bf16 \
    --target_modules model.language_model.layers.24.mlp.down_proj \
    --dataloader_num_workers 1 \
    --task_type NONE --per_device_train_batch_size 1
```


## TODO
- More Model training
- FSDP and Zero3 Support
- Demonstrate the possibility to be trained on 72B models