# Sparse Autoencoder Made Easy

<img width="3804" height="3497" alt="PixPin_2025-07-10_22-33-42" src="https://github.com/user-attachments/assets/59bf4ef7-e14c-4464-be3a-ba6fbf16ae48" />


SAE is inspired by a wealth of Sparse Autoencoder (SAE) work from Anthropic, OpenAI, Google, and the open-source community. SAE has become a powerful and widely-used tool in the field of explainable AI. 

This project aims to provide a simple and flexible interface that allows users to inject SAE modules into their models at any layer with minimal effort. We adopt the elegant design of huggingface's `peft` and regard SAE training as a kind of parameter efficient tuning, as long as the target is an `nn.Module`, SAE can be easily integrated and trained with only few lines.

## Design Philosophy
The code design takes inspiration from PEFT, as we believe SAE shares many structural similarities with PEFT-based methods. By inheriting from a BaseTuner class, we enable seamless SAE integration into existing models.

With this design, injecting an SAE module is as simple as:

```python

import torch
import torch.nn as nn
from peft import inject_adapter_in_model

from sae import TopKSaeConfig, get_peft_sae_model, PeftSaeModel

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
peft_model = get_peft_sae_model(model, config)

result = peft_model(torch.randn(1, 512, 10))
```

Loading and saving is similar to PeftModel

```python
peft_model.save_pretrained("test_save_peft_model")

model = DummyModel()
peft_model = PeftSaeModel.from_pretrained(
    model,
    "test_save_peft_model",
    adapter_name="default",
    low_cpu_mem_usage=True,
)
```

## Data Processing

To ensure consistency in data formatting, we recommend first processing your data and storing it in Parquet format. This standardization simplifies interface development and data preparation.

You are free to customize the preprocessing logic and define keys for different modalities. However, the final output should be compatible with chat templates and our preprocessing pipeline.  
An example preprocessing script is available at:  
`examples/data_process/llava_ov_clevr.py`

```sh
python examples/data_process/llava_ov_clevr.py --push_to_hub --hf_repo_path lmms-lab/LLaVA-OneVision-Data --subset "CLEVR-Math(MathV360K)" --split train --target_hf_repo_path lmms-lab/LLaVA-OneVision-Data-SAE
```

---

## Training

Our trainer implementation builds on top of existing frameworks and supports the following features:
- ZeRO-1/2/3 training
- Weights & Biases (WandB) logging

With ZeRO optimizations, you can train SAEs on 72B models using just 8×A800 GPUs.

We provide a simple training recipe to help you get started quickly. You're also welcome to implement your own training pipeline.

### Quick Start

- ZeRO-3, 72B training:  
  `examples/train/zero/run_qwen25_vl_72b_zero3.sh`

- ZeRO-2, 7B training:  
  `examples/train/zero/run_qwen25_vl_7b_zero2.sh`

- DDP, 7B training:  
  `examples/train/ddp/run_qwen25_vl_7b_ddp.sh`

### Reproducible Logs

<img width="2140" height="1830" alt="image" src="https://github.com/user-attachments/assets/2e092402-fcfb-4002-badb-55e135cd56b1" />


## Related Work and Citation
If you find this repository useful, please consider checking out our [previous paper](https://arxiv.org/pdf/2411.14982) on applying Sparse Autoencoders (SAE) to Large Multimodal Models, accepted at ICCV 2025.

You can cite our work as follows:
```shell
@misc{zhang2024largemultimodalmodelsinterpret,
      title={Large Multi-modal Models Can Interpret Features in Large Multi-modal Models},
      author={Kaichen Zhang and Yifei Shen and Bo Li and Ziwei Liu},
      year={2024},
      eprint={2411.14982},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.14982},
}
```


## Future TODOs
- [ ] Caching Sparse features to db
- [ ] Steering Sae
- [ ] Auto-Interp for Sae features
- [ ] More Sae and More training Cases!
