import math
from abc import ABCMeta
from typing import Dict, List, Tuple, Callable, Any, Optional, TypedDict

import torch
from torch import Tensor, LongTensor
from tqdm import tqdm
from transformers import StoppingCriteriaList, BatchEncoding, PreTrainedTokenizer, \
    PreTrainedModel

from nc_latent_tod.clients.abstract_lm_client import AbstractLMClient
from nc_latent_tod.utils.stop_sequence_criteria import StopSequencesStoppingCriteria


class CacheableModelInputs(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    past_key_values: Optional[Tuple[Tensor, ...]]


class AbstractHFLMClient(AbstractLMClient, metaclass=ABCMeta):
    tokenizer: PreTrainedTokenizer
    stop_sequence_criteria: StopSequencesStoppingCriteria
    max_batch_size: int
    model: PreTrainedModel
    generate_opt: Optional[Any]  # torch.compile(model) if torch_compile else None
    # prompt_text -> (common_prefix_ids, past_key_values)
    common_prefix_prompt_cache: Dict[str, Tuple[Tensor, Tuple[Tensor, ...]]]

    def store_past_kv_cache(self, common_prompt_prefix_text: str, prefix_ids: Tensor,
                            past_key_values: Tuple[Tensor, ...]):
        if self.common_prefix_prompt_cache is None:
            return
        # store the prefix portion of past_key_values for future use
        prefix_length: int = prefix_ids.shape[1]
        if past_key_values[0].shape[0] > 1:
            assert torch.allclose(past_key_values[0][0, :prefix_length - 1, :],
                                  past_key_values[0][1, :prefix_length - 1, :],
                                  # generous here due to quantization
                                  atol=1e-6), \
                "Past Key Values should correspond to the same prompt prefix across batching!"
        past_key_values = tuple(t.mean(dim=0)[None, :prefix_length, :] for t in past_key_values)
        self.common_prefix_prompt_cache[common_prompt_prefix_text] = (prefix_ids, past_key_values)

    def read_past_kv_cache(self, common_prompt_prefix_text: str) -> \
            Tuple[Optional[Tensor], Optional[Tuple[Tensor, ...]]]:
        if self.common_prefix_prompt_cache is None:
            return None, None
        return self.common_prefix_prompt_cache.get(common_prompt_prefix_text, (None, None))

    def prepare_cacheable_inputs_for_generation(self, prompt_texts: List[str],
                                                common_prompt_prefix_text: str = None,
                                                per_sequence_scale_past_kv: int = 1) \
            -> Tuple[CacheableModelInputs, LongTensor]:
        # values we need to return:
        full_input_ids: Tensor
        full_attention_mask: Tensor
        past_key_values: Optional[Tuple[Tensor, ...]]
        if common_prompt_prefix_text is None:
            # not caching! just concatenate and generate as normal
            inputs: BatchEncoding = self.tokenizer(prompt_texts, return_tensors="pt").to(self.model.device)
            return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs['attention_mask'],
                "past_key_values": None
            }, None
        else:
            prefix_ids, past_key_values = self.read_past_kv_cache(common_prompt_prefix_text)
            if prefix_ids is None:
                # need to compute the prefix_ids for the first time
                prefix_ids = self.tokenizer(common_prompt_prefix_text, return_tensors="pt").input_ids.to(
                    self.model.device)
                past_key_values = None
            # only key differences in generating with past_key_values are:
            # 1. keep and pass the past_key_values argument to the next call
            # 2. pass an attention mask that appropriately masks for concatenation of prefix and PADDED prompt text
            prefix_attn_mask: Tensor = torch.ones(prefix_ids.shape, dtype=torch.long, device=self.model.device)
            prompt_inputs: BatchEncoding = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
            prompt_ids = prompt_inputs['input_ids']
            # prompt_ids: Tensor = prompt_inputs.input_ids.to(self.model.device)
            prompt_attention_mask: Tensor = prompt_inputs['attention_mask']
            full_input_ids = torch.cat([prefix_ids.expand(prompt_ids.shape[0], -1), prompt_ids], dim=1)
            # prompt attention mask will encode our padding, which appears in the middle. Position IDs are
            # properly set by the model, using cumulative sum of attention ids.
            full_attention_mask = torch.cat(
                [prefix_attn_mask.expand(prompt_attention_mask.shape[0], -1), prompt_attention_mask], dim=1)
            if past_key_values:
                past_key_values = tuple(t.expand(prompt_ids.shape[0] * per_sequence_scale_past_kv, -1, -1)
                                        for t in past_key_values)
            return {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
                "past_key_values": past_key_values
            }, prefix_ids

    def greedy_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None,
                             max_new_tokens: int = 128) -> Dict[str, float]:
        self.model = self.model.eval()
        # I currently never actually use generate_opt, so generate_fn is always self.model.generate
        generate_fn = self.generate_opt if self.generate_opt else self.model.generate
        with torch.no_grad():
            # get cacheable inputs (may not actually be cached)
            inputs, prefix_ids = self.prepare_cacheable_inputs_for_generation([prompt_text], common_prompt_prefix_text)
            full_input_ids: Tensor = inputs['input_ids']
            past_key_values: Optional[Tuple[Tensor, ...]] = inputs['past_key_values']
            attention_mask: Tensor = inputs['attention_mask']
            # call the model
            # optional args (for some reason HF does not allow passing these as None)
            optional_args = {"past_key_values": past_key_values} if past_key_values else {}
            outputs = generate_fn(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList(
                    [self.stop_sequence_criteria.get_criteria(full_input_ids.shape[-1])]),
                **optional_args
            )
            # if we have a cache and cacheable items, store the past_key_values for future use
            if common_prompt_prefix_text and not past_key_values:
                self.store_past_kv_cache(common_prompt_prefix_text, prefix_ids, outputs.past_key_values)
            input_length: int = full_input_ids.shape[1]
            generated_tokens: Tensor = outputs.sequences[:, input_length:]
            transition_scores = self.model.compute_transition_scores(
                generated_tokens, outputs.scores, normalize_logits=True
            )

            completion, score = \
                self.stop_sequence_criteria.trim_and_score_generations(generated_tokens, transition_scores)[0]
            return {completion: score}

    def batch_greedy_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text: str = None,
                                   max_new_tokens: int = 128) -> List[Dict[str, float]]:
        self.model = self.model.eval()
        generate_fn = self.generate_opt if self.generate_opt else self.model.generate
        results: List[Dict[str, float]] = []
        sub_batches: List[List[str]] = []
        if self.max_batch_size > 0 and len(prompt_texts) > self.max_batch_size:
            # split into batches of size self.max_batch_size
            for i in range(0, len(prompt_texts), self.max_batch_size):
                sub_batches.append(prompt_texts[i:i + self.max_batch_size])
        else:
            # all in one
            sub_batches.append(prompt_texts)
        with torch.no_grad():
            for input_strings in sub_batches:
                inputs, prefix_ids = self.prepare_cacheable_inputs_for_generation(input_strings, common_prompt_prefix_text)
                full_input_ids: Tensor = inputs['input_ids']
                past_key_values: Optional[Tuple[Tensor, ...]] = inputs['past_key_values']
                attention_mask: Tensor = inputs['attention_mask']
                input_length: int = full_input_ids.shape[1]
                # optional args (for some reason HF does not allow passing these as None)
                optional_args = {"past_key_values": past_key_values} if past_key_values else {}
                outputs = generate_fn(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    stopping_criteria=StoppingCriteriaList(
                        [self.stop_sequence_criteria.get_criteria(input_length)]),
                    **optional_args
                )
                # if we have a cache and cacheable items, store the past_key_values for future use
                if common_prompt_prefix_text and not past_key_values:
                    self.store_past_kv_cache(common_prompt_prefix_text, prefix_ids, outputs.past_key_values)
                generated_tokens: Tensor = outputs.sequences[:, input_length:]
                transition_scores = self.model.compute_transition_scores(
                    generated_tokens, tuple(t.float() for t in outputs.scores), normalize_logits=True
                )
                results.extend([{k: v} for k, v in self.stop_sequence_criteria.trim_and_score_generations(
                    generated_tokens, transition_scores
                )])
        assert len(results) == len(prompt_texts), "Should have one result per prompt text"
        return results

    def top_p_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None, top_p: float = 0.9,
                            n: int = 5,
                            best_of: int = 10, max_new_tokens: int = 120,
                            temperature: float = 1.0,
                            **kwargs) -> Dict[str, float]:
        generate_fn = self.generate_opt if self.generate_opt else self.model.generate
        with torch.no_grad():
            inputs, prefix_ids = self.prepare_cacheable_inputs_for_generation([prompt_text], common_prompt_prefix_text)
            full_input_ids: Tensor = inputs['input_ids']
            attention_mask: Tensor = inputs['attention_mask']
            past_key_values: Optional[Tuple[Tensor, ...]] = inputs['past_key_values']
            input_length: int = full_input_ids.shape[1]
            result: Dict[str, float] = {}
            # generate one at a time. Slow but will work! Also works for any choice of n, best_of
            per_sample: int = max(self.max_batch_size, 1)
            num_generated = 0
            for _ in tqdm(range(0, best_of, per_sample),
                          desc=f"Generating {per_sample} top-p samples from model",
                          total=math.ceil(best_of / per_sample)):
                # doing these 1 at a time fits on a 24GB GPU AND allows us to implement stopping criteria, improving
                # speed per-generation significantly
                # optional args (for some reason HF does not allow passing these as None)
                optional_args = {"past_key_values": past_key_values} if past_key_values else {}
                outputs = generate_fn(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=per_sample,
                    temperature=temperature,
                    stopping_criteria=StoppingCriteriaList(
                       [self.stop_sequence_criteria.get_criteria(input_length)]
                    ),
                    **optional_args
                )
                # if we have a cache and cacheable items, store the past_key_values for future use
                if common_prompt_prefix_text and not past_key_values:
                    self.store_past_kv_cache(common_prompt_prefix_text, prefix_ids, outputs.past_key_values)
                generated_tokens: LongTensor = outputs.sequences[:, input_length:]
                transition_scores = self.model.compute_transition_scores(
                    generated_tokens, outputs.scores, normalize_logits=True
                )
                num_generated += generated_tokens.shape[0]
                comps_and_scores: List[Tuple[str, float]] = self.stop_sequence_criteria.trim_and_score_generations(
                    generated_tokens, transition_scores
                )
                this_sample: Dict[str, float] = dict(comps_and_scores)
                result.update(this_sample)
            assert num_generated >= best_of, "Should have one result per prompt text at least"
            return dict([(k, v) for k, v in sorted(result.items(), key=lambda pair: pair[1])[:n]])

    def batch_top_p_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text: str = None,
                                  top_p: float = 0.9, n: int = 5, best_of: int = 10, max_new_tokens: int = 120,
                                  temperature: float = 1, **kwargs) -> List[Dict[str, float]]:
        generate_fn = self.generate_opt if self.generate_opt else self.model.generate
        with torch.no_grad():
            # per_sample needs to be a multiple of len(prompt_texts)
            # per_sample also cannot exceed max_batch_size
            # per_sample needs to evenly divide expected_num_generated
            per_sample: int = max(self.max_batch_size, 1) // len(prompt_texts)

            inputs, prefix_ids = self.prepare_cacheable_inputs_for_generation(prompt_texts, common_prompt_prefix_text,
                                                                              per_sequence_scale_past_kv=per_sample)
            full_input_ids: Tensor = inputs['input_ids']
            attention_mask: Tensor = inputs['attention_mask']
            past_key_values: Optional[Tuple[Tensor, ...]] = inputs['past_key_values']
            results: List[List[Tuple[str, float]]] = [[] for _ in prompt_texts]
            expected_num_generated: int = len(prompt_texts) * best_of

            # change expected_num_generated to be a multiple of per_sample * len(prompts)
            expected_num_generated = per_sample * len(prompt_texts) * math.ceil(
                expected_num_generated / (per_sample * len(prompt_texts)))
            num_generated: int = 0

            # we pad on the left, so the 'input_length' is the same for all and is the length of the longest prompt.
            # tokenizer will trim preceding tokens (left side padding), and we won't use these in scoring.
            assert self.tokenizer.padding_side == 'left', "We assume padding is on the left in batched inference!"
            input_length: int = full_input_ids.shape[1]
            for _ in tqdm(range(0, expected_num_generated, per_sample * len(prompt_texts)),
                          desc=f"Generating {per_sample * len(prompt_texts)} top-p samples from model",
                          total=math.ceil(expected_num_generated / (per_sample * len(prompt_texts)))):
                # optional args (for some reason HF does not allow passing these as None)
                optional_args = {"past_key_values": past_key_values} if past_key_values else {}
                outputs = generate_fn(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=per_sample,
                    temperature=temperature,
                    stopping_criteria=StoppingCriteriaList(
                        [self.stop_sequence_criteria.get_criteria(input_length)]
                    ),
                    **optional_args
                )
                # if we have a cache and cacheable items, store the past_key_values for future use
                if common_prompt_prefix_text and not past_key_values:
                    self.store_past_kv_cache(common_prompt_prefix_text, prefix_ids, outputs.past_key_values)
                generated_tokens: LongTensor = outputs.sequences[:, input_length:]
                transition_scores = self.model.compute_transition_scores(
                    generated_tokens, outputs.scores, normalize_logits=True
                )
                num_generated += generated_tokens.shape[0]
                comps_and_scores: List[Tuple[str, float]] = self.stop_sequence_criteria.trim_and_score_generations(
                    generated_tokens, transition_scores
                )
                del generated_tokens
                del transition_scores
                del outputs
                torch.cuda.empty_cache()
                for start in range(0, len(comps_and_scores), per_sample):
                    end = start + per_sample
                    results[start // per_sample].extend(comps_and_scores[start:end])
            error_msg: str = f"Should have generated {expected_num_generated} sequences, got {num_generated}. " \
                             f"per_sample={per_sample}, len(prompt_texts)={len(prompt_texts)}, best_of={best_of}, " \
                             f"max_batch_size={self.max_batch_size}"
            assert num_generated == expected_num_generated, error_msg
            final_results: List[Dict[str, float]] = []
            for result in results:
                assert len(result) >= best_of, "Should have one result per prompt text at least"
                assert len(result) == len(results[0]), "Should have the same number of results for each prompt text"

                # only return the n best, sorted by score
                final_results.append(dict([(k, v) for k, v in sorted(result, key=lambda pair: pair[1])[-n:]]))
            return final_results

    def get_completion_log_probabilities(self, prompt_text: str, completion: str, common_prompt_prefix_text: str = None,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[
        float]:
        log_probs: List[List[float]] = self.batch_get_completion_log_probabilities(
            [prompt_text], [completion], common_prompt_prefix_text=common_prompt_prefix_text
        )
        if token_log_probs_telemetry_hook:
            token_log_probs_telemetry_hook(log_probs[0])
        return log_probs[0]

    def batch_get_completion_log_probabilities(self, prompt_texts: List[str], completions: List[str],
                                               common_prompt_prefix_text: str = None,
                                               token_log_probs_telemetry_hook: Callable[
                                                   [List[List[float]]], None] = None) -> List[List[float]]:
        sub_batches: List[Tuple[List[str], List[str]]] = []
        if self.max_batch_size > 0 and len(prompt_texts) > self.max_batch_size:
            # split into batches of size self.max_batch_size
            for i in range(0, len(prompt_texts), self.max_batch_size):
                sub_batches.append((prompt_texts[i:i + self.max_batch_size], completions[i:i + self.max_batch_size]))
        else:
            # all in one
            sub_batches.append((prompt_texts, completions))

        full_result: List[List[float]] = []
        with torch.no_grad():
            prefix_prompt_ids: Optional[Tensor] = None
            past_key_values: Optional[Tuple[Tensor, ...]] = None
            if common_prompt_prefix_text:
                prefix_prompt_ids, past_key_values = self.read_past_kv_cache(common_prompt_prefix_text)
                if prefix_prompt_ids is None:
                    # Note: on CPU!
                    prefix_prompt_ids = self.tokenizer(common_prompt_prefix_text, return_tensors="pt").input_ids
                    past_key_values = None
                else:
                    # we're working on CPU as we build the inputs
                    prefix_prompt_ids = prefix_prompt_ids.cpu()
            for batch_texts, batch_completions in sub_batches:
                prompt_tensors: List[Tensor] = []
                completion_tensors: List[Tensor] = []
                full_tensors: List[Tensor] = []  # prompts concatenated with completions
                for prompt_text, completion in zip(batch_texts, batch_completions, strict=True):
                    prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
                    if prefix_prompt_ids is not None:
                        prompt_ids = torch.cat([prefix_prompt_ids, prompt_ids], dim=1)
                    completion_ids = self.tokenizer(completion, return_tensors="pt").input_ids
                    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    prompt_tensors.append(prompt_ids)
                    completion_tensors.append(completion_ids)
                    full_tensors.append(full_ids.squeeze())
                min_prompt_length: int = min([x.shape[1] for x in prompt_tensors])
                past_kv_length: int = past_key_values[0].shape[1] if past_key_values else 0

                # prepare a batch from full_tensors
                x: Tensor = torch.nn.utils.rnn.pad_sequence(full_tensors, batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id).to(self.model.device)

                if past_key_values:
                    past_key_values = tuple(t.expand(x.shape[0], -1, -1) for t in past_key_values)
                # attention_mask is where we mask out the padding tokens
                inputs = self.model.prepare_inputs_for_generation(x, **{
                    "attention_mask": (x != self.tokenizer.pad_token_id).long(),
                    "past_key_values": past_key_values if past_key_values else None,
                })
                logits = self.model(**inputs).logits
                shift_logits = logits[..., min_prompt_length - past_kv_length - 1:-1, :].contiguous()
                shift_labels = x[..., min_prompt_length:].contiguous()
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                         shift_labels.view(-1), reduction='none')
                log_probs = -loss.view(shift_labels.size())
                for prompt_ids, completion_ids, seq_log_probs in zip(prompt_tensors, completion_tensors, log_probs, strict=True):
                    start: int = prompt_ids.shape[1] - min_prompt_length
                    end: int = start + completion_ids.shape[1]
                    full_result.append(seq_log_probs[start:end].cpu().squeeze().tolist())
        if token_log_probs_telemetry_hook:
            token_log_probs_telemetry_hook(full_result)
        return full_result

    def wait_until_model_ready(self, timeout_in_seconds: int = 60) -> bool:
        # ready on __init__
        return True
