import os
import time
from pathlib import Path
from typing import Union, Callable, Dict, Any

import torch
from peft import set_peft_model_state_dict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class TFLOPSCallback(TrainerCallback):
    """
    A callback which computes average achieved TFLOPs over the course of a training run: toal floating point ops
    divided by total training time.
    """
    train_step_start: float
    total_train_time: float
    total_train_samples: int
    total_train_steps: int
    get_time: Callable[[], float]
    logging_callback: Callable[[Dict[str, Any]], None]

    def __init__(self, logging_callback: Callable[[Dict[str, Any]], None] = print) -> None:
        """
        Instantiate the call-back with a logging mechanism (by default it will just print). An example for logging to
        wandb:

        >>> callback = TFLOPSCallback(logging_callback=wandb.log)

        :param logging_callback: callback which takes a dictionary of metrics and logs them in a meaningful way (e.g.
            `wandb.log`)
        """
        super().__init__()
        self.logging_callback = logging_callback
        self.train_step_start = -1
        self.total_train_time = 0
        self.total_train_samples = 0
        self.total_train_steps = 0
        # time.time() will be inaccurate and subject to factor's beyond our control, but will handle process
        # switching, etc. Should be ok when averaged.
        self.get_time = time.time

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_begin(args, state, control, **kwargs)
        self.train_step_start = self.get_time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        self.total_train_time += self.get_time() - self.train_step_start
        self.total_train_steps += 1
        self.total_train_samples += args.train_batch_size

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        total_flops: float = state.total_flos
        self.logging_callback({
            "train/achieved_tflops": (total_flops / self.total_train_time) / 1e12,
            "train/time_in_train_steps": self.total_train_time,
            "train/my_samples_per_second": self.total_train_samples / self.total_train_time,
            "train/my_steps_per_second": self.total_train_steps / self.total_train_time,
        })

# Adapted From Huggingface examples
class SavePeftModelCallback(TrainerCallback):
    """
    A call back which can be used with Huggingface Trainer to save a PEFT model
    """

    checkpoint_dir: Union[str, Path]

    def __init__(self, checkpoint_dir: Union[str, Path] = "./checkpoints") -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder: str = os.path.join(self.checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        # save nothing for the main model
        torch.save({}, pytorch_model_path)
        return control


# Adapted From Huggingface examples
class LoadBestPeftModelCallback(TrainerCallback):
    """
    A call-back for loading the best Peft Model from the saved checkpoints
    """
    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def print_trainable_parameters(model) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )