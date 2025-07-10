import unittest

import torch
import torch.nn as nn
from sae import TopKSaeConfig, get_peft_sae_model


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestForward(unittest.TestCase):
    def test_train_peft_model(self):
        model = DummyModel()
        config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

        model.train()  # Set the model to training mode

        tensor_input = torch.randn(1, 512, 10)
        original_result = model(tensor_input)

        # Get the PEFT model
        peft_model = get_peft_sae_model(model, config)

        result = peft_model(tensor_input)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 512, 10))
        torch.testing.assert_close(original_result, result)

    def test_eval_peft_model(self):
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

        result = peft_model(tensor_input)
        input_hidden_dict = peft_model.base_model.input_hidden_dict
        output_hidden_dict = peft_model.base_model.output_hidden_dict

        sae_in = input_hidden_dict["model.linear"]
        sae_out = output_hidden_dict["model.linear"]

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 512, 10))
        torch.testing.assert_close(original_result.flatten(0, 1), sae_in)
        torch.testing.assert_close(result.flatten(0, 1), sae_out)


if __name__ == "__main__":
    unittest.main()
