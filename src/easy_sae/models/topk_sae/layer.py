import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import TopKSaeConfig


class TopKSaeLayer(BaseTunerLayer):
    """
    Sparse Autoencoder Layer for PEFT.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("sae_encoder", "sae_W_dec", "sae_b_dec")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("k",)

    def __init__(self, base_layer: nn.Module, *args, **kwargs):
        self.base_layer = base_layer
        self.sae_encoder = nn.ModuleDict({})
        self.sae_W_dec = nn.ParameterDict({})
        self.sae_b_dec = nn.ParameterDict({})
        self.k = {}

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(
                base_layer, "out_features"
            ):
                in_features, out_features = (
                    base_layer.in_features,
                    base_layer.out_features,
                )
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.",
                UserWarning,
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        k: int,
        num_latents: int,
        expansion_factor: int,
        **kwargs,
    ):
        if num_latents == 0:
            num_latents = expansion_factor * self.in_features

        self.sae_encoder[adapter_name] = nn.Linear(
            self.out_features, num_latents, bias=False
        )
        # When we init new adapters, we copy the weights from the encoder for decoder
        self.sae_W_dec[adapter_name] = nn.Parameter(
            self.sae_encoder[adapter_name].weight.data.clone()
        )
        self.sae_b_dec[adapter_name] = nn.Parameter(
            torch.zeros(self.out_features, dtype=torch.float32)
        )
        self.k[adapter_name] = k
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        assert False, "Merging is not supported for Linear layers in SAE."

    def unmerge(self):
        assert False, "Unmerging is not supported for Linear layers in SAE."

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)


class Linear(nn.Module, TopKSaeLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str = None,
        k: int = 32,
        num_latents: int = 0,
        expansion_factor: int = 32,
        **kwargs,
    ):
        super().__init__()
        TopKSaeLayer.__init__(self, base_layer, **kwargs)

        self._activate_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            k=k,
            num_latents=num_latents,
            expansion_factor=expansion_factor,
        )

    def eager_decode(
        self, top_indices: torch.Tensor, top_acts: torch.Tensor, W_dec: torch.Tensor
    ):
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        return acts @ W_dec.mT

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
            final_result = result
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            # If we use sae, sae is reconstruction, we add and avg multiple saes
            final_result = torch.zeros_like(result, dtype=torch_result_dtype)

            for activate_adapter in self.active_adapters:
                encoder = self.sae_encoder[activate_adapter]
                W_dec = self.sae_W_dec[activate_adapter]
                bias = self.sae_b_dec[activate_adapter]
                k = self.k[activate_adapter]
                result = self._cast_input_dtype(result, encoder.weight.dtype)
                # Remove decoder bias as per Anthropic
                sae_in = result - bias
                pre_act: torch.Tensor = encoder(sae_in)
                top_acts, top_indices = pre_act.topk(k, sorted=False)
                sae_out = self.eager_decode(top_indices, top_acts, W_dec.mT)
                sae_out = sae_out + bias
                final_result += sae_out

            final_result /= len(self.active_adapters)
            final_result = final_result.to(torch_result_dtype)

        return final_result

    def __repr__(self):
        rep = super().__repr__()
        return "sae." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    topk_sae_config: TopKSaeConfig,
    **kwargs,
):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Linear):
        new_module = Linear(
            base_layer=target_base_layer,
            adapter_name=adapter_name,
            **kwargs,
        )

    return new_module
