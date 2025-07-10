import unittest

import torch
import torch.nn as nn
from sae import PeftSaeModel, TopKSaeConfig, get_peft_sae_model


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestSaveAndLoad(unittest.TestCase):
    def test_save_and_load_peft_model(self):
        # When in evaluation mode, we will expect that the sae reconstruction will be return to the input stream
        # So the output should be the same as the output hidden dict, but input hidden dict should be the same as the input
        model = DummyModel()
        config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

        model.eval()  # Set the model to evaluation mode

        tensor_input = torch.randn(1, 512, 10)
        original_result = model(tensor_input)

        # Get the PEFT model
        peft_model = get_peft_sae_model(model, config)
        peft_model.eval()

        peft_model.save_pretrained("test_save_peft_model")

        model = DummyModel()
        peft_model = PeftSaeModel.from_pretrained(
            model,
            "test_save_peft_model",
            adapter_name="default",
            low_cpu_mem_usage=True,
        )
        # Remove the local directory after saving
        import shutil

        shutil.rmtree("test_save_peft_model")


if __name__ == "__main__":
    unittest.main()
