from transformers import Trainer


class SaeTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)

        # Possibly wrap up here, we use the original unwrapped model
        output_hidden_dict = self.model.base_model.output_hidden_dict
        input_hidden_dict = self.model.base_model.input_hidden_dict

        per_layer_loss = {}
        total_loss = 0

        for layer, output_hidden_states in output_hidden_dict.items():
            input_hidden_states = input_hidden_dict[layer]
            e = output_hidden_states - input_hidden_states
            total_variance = (
                (input_hidden_states - input_hidden_states.mean(0)).pow(2).sum()
            )
            l2_loss = e.pow(2).sum()
            fvu = l2_loss / total_variance
            per_layer_loss[layer] = fvu.item()
            total_loss += fvu

        self.log(per_layer_loss)

        return (total_loss, outputs) if return_outputs else total_loss
