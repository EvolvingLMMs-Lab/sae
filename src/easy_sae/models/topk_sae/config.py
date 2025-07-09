from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

from ...utils import SaeType


@dataclass
class TopKSaeConfig(PeftConfig):
    """
    Configuration for training a sparse autoencoder on a language model.

    Args:
        expansion_factor (int): Multiple of the input dimension to use as the SAE dimension.
        num_latents (int): Number of latents to use. If 0, use `expansion_factor`.
        k (int): Number of nonzero features.
    """

    expansion_factor: int = field(
        default=32,
        metadata={
            "help": "Multiple of the input dimension to use as the SAE dimension."
        },
    )
    num_latents: int = field(
        default=0,
        metadata={"help": "Number of latents to use. If 0, use `expansion_factor`."},
    )
    k: int = field(default=32, metadata={"help": "Number of nonzero features."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "list of module names or regex expression of the module names to replace with RandLora."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    dead_tokens_threshold: Optional[int] = field(
        default=10000000,
        metadata={
            "help": (
                "Threshold for dead tokens. If the number of tokens fired is less than this threshold, "
                "the token is considered dead."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = SaeType.TOPK_SAE
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )
