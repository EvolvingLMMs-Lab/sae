from typing import Any

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)
from peft.utils import AuxiliaryTrainingWrapper

from .config import TopKSaeConfig
from .layer import TopKSaeLayer, dispatch_default


class TopKSaeModel(BaseTuner):
    prefix: str = "sae"

    def __init__(self, model, peft_config, adapter_name, low_cpu_mem_usage=False):
        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage)

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    def _create_and_replace(
        self,
        peft_config: TopKSaeConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **kwargs: Any,
    ):
        kwargs.update(
            {
                "k": peft_config.k,
                "num_latents": peft_config.num_latents,
                "expansion_factor": peft_config.expansion_factor,
            }
        )
        if isinstance(target, TopKSaeLayer):
            # If the target is already a TopKSaeLayer, we can update it directly
            target.update_layer(
                adapter_name=adapter_name,
                k=peft_config.k,
                num_latents=peft_config.num_latents,
                expansion_factor=peft_config.expansion_factor,
            )
        else:
            device_map = (
                self.model.hf_device_map
                if hasattr(self.model, "hf_device_map")
                else None
            )
            new_module = self._create_new_module(
                peft_config, adapter_name, target, device_map=device_map, **kwargs
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        peft_config: TopKSaeConfig, adapter_name: str, target: nn.Module, **kwargs: Any
    ) -> nn.Module:
        dispatchers = []

        dispatchers.extend([dispatch_default])
        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(
                target, adapter_name, topk_sae_config=peft_config, **kwargs
            )
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`"
            )

        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _check_target_module_exists(sae_config, key):
        return check_target_module_exists(sae_config, key)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, TopKSaeLayer):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)
