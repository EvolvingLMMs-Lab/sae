from abc import abstractmethod
from typing import Dict

import torch
from peft.tuners.tuners_utils import BaseTuner


class BaseSaeModel(BaseTuner):
    prefix: str = "sae_"

    def __init__(self, model, peft_config, adapter_name, low_cpu_mem_usage=False):
        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage)
        self.output_hidden_dict: Dict[str, torch.Tensor] = {}
        self.input_hidden_dict: Dict[str, torch.Tensor] = {}

    @property
    def prepare_inputs_for_generation(self):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation
        else:
            return None

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    def get_aux_log_info():
        """
        Returns a dictionary containing auxiliary information for logging.
        """
        return {}
