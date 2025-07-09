from typing import Any, Dict

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING


class SaeFactory:
    @staticmethod
    def sae_config_mapping(sae_type: str):
        """
        Maps the SAE type to its corresponding configuration class.

        Args:
            sae_type (str): The type of SAE.

        Returns:
            PeftConfig: The configuration class for the specified SAE type.

        Raises:
            NotImplementedError: If the SAE type is not implemented.
        """
        sae_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(sae_type, None)
        if sae_config_cls is None:
            raise NotImplementedError(f"Sae type : {sae_type} not implemented yet")
        return sae_config_cls

    @staticmethod
    def create_sae_config(sae_type: str, sae_config: Dict[str, Any], **kwargs):
        """
        Creates an SAE configuration instance based on the SAE type and additional parameters.

        Args:
            sae_type (str): The type of SAE.
            **kwargs: Additional parameters for the SAE configuration.

        Returns:
            PeftConfig: An instance of the SAE configuration class.
        """
        sae_config_cls = SaeFactory.sae_config_mapping(sae_type)
        if sae_type == "TOPK_SAE":
            sae = sae_config_cls(
                num_latents=sae_config.get("num_latents", 4096),
                k=sae_config.get("k", 32),
                target_modules=sae_config.get("target_modules", None),
                dead_tokens_threshold=sae_config.get("dead_tokens_threshold", 10000000),
            )
        else:
            raise NotImplementedError(f"Sae type : {sae_type} not implemented yet")

        return sae
