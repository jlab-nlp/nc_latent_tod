from typing import List, Optional, Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, \
    PreTrainedTokenizer

from nc_latent_tod.clients.abstract_hf_lm_client import AbstractHFLMClient
from nc_latent_tod.utils.stop_sequence_criteria import StopSequencesStoppingCriteria


class StarCoderClient(AbstractHFLMClient):
    model: PreTrainedModel
    generate_opt: Optional[Any]
    tokenizer: PreTrainedTokenizer
    stop_sequences: List[str]
    stop_sequence_criteria: StopSequencesStoppingCriteria
    max_batch_size: int

    def __init__(self, stop_sequences: List[str] = None, model_name_or_path: str = "bigcode/starcoder",
                 adapter_path: Optional[str] = None, model: PreTrainedModel = None,
                 max_batch_size: int = -1,
                 load_in_8bit: bool = True,
                 torch_compile: bool = False,
                 use_past_key_value_cache: bool = False,
                 **kwargs) -> None:
        model_name_or_path = model_name_or_path or "bigcode/starcoder"
        self.stop_sequences = stop_sequences or ['#', 'print(', 'print("', '\n    \n    ']
        self.max_batch_size = max_batch_size
        if '<|endoftext|>' not in self.stop_sequences:
            self.stop_sequences.append('<|endoftext|>')
        if not model:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, load_in_8bit=torch.cuda.is_available() and load_in_8bit,
                use_auth_token=True, device_map="auto",
                **kwargs
            )
        else:
            self.model = model
        if torch_compile:
            self.model = self.model.eval()
            self.generate_opt = torch.compile(self.model.generate)
        else:
            self.generate_opt = None
            # logging.warning("Torch compile is not yet implemented!")
        if adapter_path:
            self.model: PeftModel = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Using adapted model: {type(self.model)}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        # this will terminate generation early on our stop sequences
        self.stop_sequence_criteria = StopSequencesStoppingCriteria(self.stop_sequences, self.tokenizer)
        self.common_prefix_prompt_cache = {} if use_past_key_value_cache else None
