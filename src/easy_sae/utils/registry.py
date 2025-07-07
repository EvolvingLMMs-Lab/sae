from typing import Optional


def register_sae_method(
    *,
    name: str,
    config_cls,
    model_cls,
    prefix: Optional[str] = None,
    is_mixed_compatible=False,
    peft_type: str = None,
) -> None:
    """
    Function to register a finetuning method like LoRA to be available in PEFT.

    This method takes care of registering the PEFT method's configuration class, the model class, and optionally the
    prefix.

    Args:
        name (str):
            The name of the PEFT method. It must be unique.
        config_cls:
            The configuration class of the PEFT method.
        model_cls:
            The model class of the PEFT method.
        prefix (Optional[str], optional):
            The prefix of the PEFT method. It should be unique. If not provided, the name of the PEFT method is used as
            the prefix.
        is_mixed_compatible (bool, optional):
            Whether the PEFT method is compatible with `PeftMixedModel`. If you're not sure, leave it as False
            (default).
        peft_type (str, optional):
            The type of the PEFT method. If not provided, it will be set to the mapping manually

    Example:

        ```py
        # inside of peft/tuners/my_peft_method/__init__.py
        from peft.utils import register_peft_method

        register_sae_method(name="my_peft_method", config_cls=MyConfig, model_cls=MyModel, peft_type="MY_PEFT_METHOD")
        ```
    """
    from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )

    if name.endswith("_"):
        raise ValueError(
            f"Please pass the name of the PEFT method without '_' suffix, got {name}."
        )

    if not name.islower():
        raise ValueError(
            f"The name of the PEFT method should be in lower case letters, got {name}."
        )

    # model_cls can be None for prompt learning methods, which don't have dedicated model classes
    if prefix is None:
        prefix = name + "_"

    if (
        (peft_type in PEFT_TYPE_TO_CONFIG_MAPPING)
        or (peft_type in PEFT_TYPE_TO_TUNER_MAPPING)
        or (peft_type in PEFT_TYPE_TO_MIXED_MODEL_MAPPING)
    ):
        raise KeyError(
            f"There is already PEFT method called '{name}', please choose a unique name."
        )

    if prefix in PEFT_TYPE_TO_PREFIX_MAPPING:
        raise KeyError(
            f"There is already a prefix called '{prefix}', please choose a unique prefix."
        )

    model_cls_prefix = getattr(model_cls, "prefix", None)
    if (model_cls_prefix is not None) and (model_cls_prefix != prefix):
        raise ValueError(
            f"Inconsistent prefixes found: '{prefix}' and '{model_cls_prefix}' (they should be the same)."
        )

    PEFT_TYPE_TO_PREFIX_MAPPING[peft_type] = prefix
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls
    if is_mixed_compatible:
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING[peft_type] = model_cls
