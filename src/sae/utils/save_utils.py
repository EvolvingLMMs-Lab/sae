from peft.utils.peft_types import PeftType
from peft.utils.save_and_load import (
    get_peft_model_state_dict as peft_get_peft_model_state_dict,
)

from .sae_type import SaeType


def get_peft_model_state_dict(
    model,
    state_dict=None,
    adapter_name="default",
    unwrap_compiled=False,
    save_embedding_layers="auto",
):
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in PeftType:
        return peft_get_peft_model_state_dict(
            model=model,
            state_dict=state_dict,
            adapter_name=adapter_name,
            unwrap_compiled=unwrap_compiled,
            save_embedding_layers=save_embedding_layers,
        )

    if config.peft_type == SaeType.TOPK_SAE:
        to_return = {k: state_dict[k] for k in state_dict if "sae_" in k}
    else:
        raise ValueError(f"Unsupported PEFT type: {config.peft_type}")

    # REMOVE ADAPTER NAME
    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return
