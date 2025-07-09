import unittest

import torch
import torch.nn as nn
from peft import inject_adapter_in_model

from easy_sae import TopKSaeConfig, get_peft_sae_model


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestInjectAdapter(unittest.TestCase):
    def test_inject_adapter(self):
        model = DummyModel()
        config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

        # Inject the adapter into the model
        model = inject_adapter_in_model(config, model)

        # Check if the adapter was injected correctly
        result = model(torch.randn(1, 512, 10))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 512, 10))

    def test_get_peft_model(self):
        model = DummyModel()
        config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

        # Get the PEFT model
        peft_model = get_peft_sae_model(model, config)

        result = peft_model(torch.randn(1, 512, 10))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 512, 10))


if __name__ == "__main__":
    unittest.main()
