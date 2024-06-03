from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from nc_latent_tod.experiments.config import DataConfigDC



@dataclass
class StarcoderModelConfig:
    """
    Fields:
    - model_name_or_path (str): name of the starcoder model to fine-tune (default = "bigcode/starcoder")
    - lora_alpha (`int`): The alpha parameter for Lora scaling. See peft.LoraConfig.lora_alpha
    - lora_r (`int`): Lora attention dimension. See peft.LoraConfig.r
    """
    model_name_or_path: Optional[str] = "bigcode/starcoder"
    lora_alpha: Optional[int] = 32
    lora_r: Optional[int] = 16
    stop_sequences: Optional[List[str]] = None


@dataclass
class TrainerConfig:
    other_trainer_arguments: Optional[Dict[str, Any]]
    max_steps: Optional[int] = 10_000
    eval_steps: Optional[int] = 1_000
    save_steps: Optional[int] = 1_000
    gradient_accumulation_steps: Optional[int] = 1
    per_device_train_batch_size: Optional[int] = 8
    per_device_eval_batch_size: Optional[int] = 16
    memory_metrics: Optional[bool] = False
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False


@dataclass
class DataConfigWithRetrieval(DataConfigDC):
    retriever_model_name_or_path: Optional[str]
    index_set_path_or_name: Optional[str]
    index_set_split_name: Optional[str]
    retriever_from_artifact: Optional[bool]
    num_examples: Optional[int] = 5


@dataclass
class FinetuneStarcoderConfig:
    """
    Fields:
    - model (`StarcoderModelConfig`): configuration for model to fine-tune
    """
    model_cfg: StarcoderModelConfig
    data_cfg: DataConfigWithRetrieval
    output_dir: Optional[str]
    training_cfg: TrainerConfig
