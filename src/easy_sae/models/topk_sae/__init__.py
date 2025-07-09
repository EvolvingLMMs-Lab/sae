from easy_sae.utils.registry import register_sae_method

from ...utils import SaeType
from .config import TopKSaeConfig
from .layer import TopKSaeLayer
from .model import TopKSaeModel

register_sae_method(
    name="topk_sae",
    config_cls=TopKSaeConfig,
    model_cls=TopKSaeModel,
    prefix=TopKSaeModel.prefix,
    is_mixed_compatible=True,
    peft_type=SaeType.TOPK_SAE,
)

__all__ = [
    "TopKSaeConfig",
    "TopKSaeModel",
    "TopKSaeLayer",
]
