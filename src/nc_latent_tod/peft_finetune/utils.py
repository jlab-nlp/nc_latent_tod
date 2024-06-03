from typing import Optional, Literal, List, Dict

import torch
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, BatchEncoding

from nc_latent_tod.peft_finetune.config import StarcoderModelConfig
from nc_latent_tod.peft_finetune.data import LMDataInstance


def build_peft_starcoder_model(model_cfg: StarcoderModelConfig) -> PeftModel:
    """
    Given a configuration for a Starcoder model to fine-tune, build and return to corresponding PeftModel
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_cfg.model_name_or_path,
        use_auth_token=True,
        use_cache=True,
        # note this argument for loading the in 8-bit mode
        load_in_8bit=True,
        device_map="auto",
    )

    # some model preparation work done by `peft`
    model = prepare_model_for_kbit_training(model)

    # For our parameter efficient tuning method, we'll use LoRA
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"]
    )

    # get a peft model based on our config and base model
    model: PeftModel = get_peft_model(model, lora_config)
    return model


LabelMaskMode = Literal["padding", "prompt", "none"]


def prompt_and_completion_to_inputs(tokenizer: PreTrainedTokenizerBase, instances: LMDataInstance,
                                    label_mask_mode: Optional[LabelMaskMode] = None) -> BatchEncoding:
    """
    Given an LMDataInstance (or batched LM data instances, w/ same keys and value lists), returns a BatchEncoding
    with the following keys:
    - input_ids: the input ids for the model (padding + prompt + completion)
    - attention_mask: the attention mask for the model (padding + prompt + completion)
    - labels: the labels for the model, masked according to label_mask_mode. If label_mask_mode is None, then
      no masking is applied and labels == input_ids. If label_mask_mode is "padding", then padding tokens are masked to
      -100, and the remainder are left as equal to input_ids. If label_mask_mode is "prompt", then padding and prompt
      tokens are masked to -100, and only the completion tokens are left equal to input_ids. This allows for controlled
      training setups, where cross-entropy loss is only computed on part of the sequence.

    :param tokenizer: the tokenizer to use (needs a pad token, and needs to be left padding if using label_mask_mode)
    :param instances: the LMDataInstance or batched LMDataInstances to convert to inputs
    :param label_mask_mode: the label mask mode to use, or None for no masking. See above for details.
    """
    # tokenize prompts + completions batched, as one sequence
    # special case: if we call with str, str to code-llama tokenizer, it thinks we are in-filling and adds special
    # tokens, which aren't available in code-llama 7B and are for the wrong task. Calling with a list supports batches
    # of size 1.
    prompts: List[str] = instances["prompt"] \
        if type(instances["prompt"]) == list \
        else [instances["prompt"]]
    completions: List[str] = instances["completion"] \
        if type(instances["completion"]) == list \
        else [instances["completion"]]
    full_texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
    text_inputs = tokenizer(
        full_texts,
        return_tensors='pt',
        padding=True,
        return_offsets_mapping=True
    )

    input_ids: torch.Tensor = text_inputs["input_ids"]

    # start with identical labels (model shifts for us)
    labels: torch.Tensor = input_ids.clone()
    if label_mask_mode and label_mask_mode != "none":
        # we'll mark tokens as non loss applicable if they are padding (or prompt, when configured)
        # 1 for every non padding token, so length minus sum gives number of inserted <pad> tokens
        padding_lengths = input_ids.shape[1] - text_inputs["attention_mask"].sum(dim=-1)
        if padding_lengths.sum() > 0:
            assert tokenizer.padding_side == 'left', "we assume padding is on the left if padding is present"
        if label_mask_mode == "prompt":
            # Get prompt lengths using offsets
            prompt_char_lengths: Tensor = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
            furthest_start: Tensor = text_inputs["offset_mapping"].max(dim=-1).values
            # a token is a prompt token if its offset end (max) is less than the length of the prompt
            num_prompt_tokens: Tensor = (furthest_start < prompt_char_lengths[:, None]).sum(dim=-1)
            mask_lengths: torch.Tensor = num_prompt_tokens
        elif label_mask_mode == "padding":
            mask_lengths = padding_lengths

        # set labels to -100 for all padding & prompt tokens
        mask = torch.less(torch.arange(labels.shape[-1]), mask_lengths[:, None])
        labels[mask] = -100

    text_inputs['labels'] = labels

    return BatchEncoding({
        "input_ids": input_ids.squeeze(),
        "attention_mask": text_inputs["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    })