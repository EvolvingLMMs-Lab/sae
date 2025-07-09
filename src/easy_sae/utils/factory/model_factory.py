import importlib

import transformers


class ModelFactory:
    @staticmethod
    def create_model(model_name: str, **kwargs):
        # TODO : Add support for other model types in the future, for example, custom model
        try:
            # Attempt to create a Hugging Face model using the provided model name
            return ModelFactory.create_hf_model(model_name, **kwargs)
        except Exception as e:
            # If it fails, raise an error with a clear message
            raise ValueError(
                f"Failed to create model for {model_name}. Error: {str(e)}"
            )

    @staticmethod
    def create_hf_model(
        model_name: str, torch_dtype, attn_implementation: str = "sdpa", **kwargs
    ) -> transformers.PreTrainedModel:
        """
        Create a Hugging Face model for the given model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            transformers.PreTrainedModel: The model for the given name.
        """
        config = transformers.AutoConfig.from_pretrained(model_name)

        model_arch_name = (
            config.architectures[0] if hasattr(config, "architectures") else None
        )
        if model_arch_name is None:
            raise ValueError(
                f"Model {model_name} does not have a valid architecture defined in its config."
            )

        # Dynamically import the model class based on the architecture name
        transformers_module = importlib.import_module("transformers")
        model_class = getattr(transformers_module, model_arch_name, None)
        if model_class is None:
            model_class = (
                transformers.AutoModelForCausalLM
                if "CausalLM" in model_arch_name
                else transformers.AutoModel
            )

        model = model_class.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        return model
