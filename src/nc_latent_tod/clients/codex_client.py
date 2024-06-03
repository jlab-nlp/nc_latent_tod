"""
Methods for generating language model completions with the Codex family of models via the OpenAI API.

This file was adapted from the code for the paper "In Context Learning for Dialogue State Tracking", as originally
published here: https://github.com/Yushi-Hu/IC-DST. Cite their article as:

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
"""
import logging
import os
from typing import List, TypedDict, Optional, Dict, Any, Callable

import openai
from openai.error import RateLimitError, APIError, APIConnectionError, OpenAIError, InvalidRequestError

from nc_latent_tod.clients.abstract_lm_client import AbstractLMClient
from nc_latent_tod.utils.general import check_argument
from nc_latent_tod.utils.speed_limit_timer import SpeedLimitTimer

TOO_MANY_TOKENS_FOR_ENGINE: str = "This model's maximum context length is"


class PromptOverlengthError(ValueError):
    """
    A ValueError specific to the case where the prompt is longer than the permitted number of tokens
    """
    pass


class OpenAIAPIConfig(TypedDict):
    """
    A dictionary of config items for OpenAI API use
    """
    api_key: str
    organization: Optional[str]  # optional to toggle between a chosen one and API key default
    seconds_per_step: float


def _load_environment_codex_config() -> OpenAIAPIConfig:
    api_key: str = os.environ.get("OPENAI_API_KEY_JLAB_ORG") or os.environ.get("OPENAI_API_KEY")
    organization: str = os.environ.get("OPENAI_ORGANIZATION")
    check_argument(api_key, "must set an API key. Use environment variable OPENAI_API_KEY or otherwise provide "
                            "a CodexConfig")
    return {"api_key": api_key.strip(),  # easier than fixing a k8s secret
            "organization": organization,
            "seconds_per_step": 0.25}


class CodexClient(AbstractLMClient):
    """
    Simplified client for working with Codex and OpenAI models, wraps openai client.
    """

    config: OpenAIAPIConfig
    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer

    def __init__(self, config: OpenAIAPIConfig = None, engine: str = "gpt-3.5-turbo-instruct-0914",
                 stop_sequences: List[str] = None) -> None:
        super().__init__()
        self.config = config or _load_environment_codex_config()
        self.engine = engine
        self.stop_sequences = stop_sequences or ['--', '\n', ';', '#']
        self.timer = SpeedLimitTimer(second_per_step=self.config['seconds_per_step'])  # openai limitation 20 query/min

    def greedy_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :param common_prompt_prefix_text: prefix text for OpenAI Completion API. If specified, the true prompt to the LM
            will be the concatenation of common_prompt_prefix_text and prompt_text.
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            if common_prompt_prefix_text:
                # nothing fancy for OpenAI models, just concatenate the strings
                prompt_text = common_prompt_prefix_text + prompt_text
            args: Dict[str, Any] = {
                "engine": self.engine,
                "prompt": prompt_text,
                "max_tokens": 120,
                "logprobs": 1,
                "temperature": 0,
                "stop": stop_sequences,
                "organization": self.config.get('organization', openai.organization),
                "api_key": self.config.get('api_key', openai.api_key)
            }

            self.timer.step()
            result = openai.Completion.create(**args)
            completions = dict(zip(
                [x['text'] for x in result['choices']],
                [sum(x['logprobs']['token_logprobs']) for x in result['choices']]
            ))
            return completions
        except InvalidRequestError as e:
            if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
                raise PromptOverlengthError(e)
            else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(8)
            raise e

    def batch_greedy_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text=None,
                                   max_new_tokens: int = 128) -> List[Dict[str, float]]:
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            args: Dict[str, Any] = {
                "engine": self.engine,
                "prompt": prompt_texts,
                "max_tokens": 120,
                "logprobs": 1,
                "temperature": 0.,
                "stop": stop_sequences,
                "organization": self.config.get('organization', openai.organization),
                "api_key": self.config.get('api_key', openai.api_key)
            }

            self.timer.step()
            batch_codex_result = openai.Completion.create(**args)
            our_batch_result: List[Dict[str, float]] = []
            # in greedy mode, we can assume 1 choice per batch item
            assert len(batch_codex_result['choices']) == len(prompt_texts)
            for result in batch_codex_result['choices']:
                our_batch_result.append({result['text']: sum(result['logprobs']['token_logprobs'])})
            return our_batch_result
        except InvalidRequestError as e:
            if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
                raise PromptOverlengthError(e)
            else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def top_p_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None, top_p: float = 0.9,
                            n: int = 5, best_of: int = 10, max_new_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for OpenAI Completion API
        :param common_prompt_prefix_text: prefix text for OpenAI Completion API. If specified, the true prompt to the LM
            will be the concatenation of common_prompt_prefix_text and prompt_text.
        :param top_p: the nucleus sampling parameter
        :param n: max number of completions to include in the final result
        :param best_of: number of completions to sample and return the n best of
        :param max_new_tokens: maximum number of tokens to generate per sample
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            # nothing fancy for OpenAI models, just concatenate the strings
            prompt_text = common_prompt_prefix_text + prompt_text if common_prompt_prefix_text else prompt_text
            args = {
                "engine": self.engine,
                "prompt": prompt_text,
                "max_tokens": max_new_tokens,
                "top_p": top_p,
                "stop": stop_sequences,
                "n": n,
                "logprobs": 1,  # 1 needed to get back log-probabilities at all, in choice['logprobs']['token_logprobs']
                "best_of": best_of,
            }
            self.timer.step(best_of * self.config['seconds_per_step'])
            result = openai.Completion.create(**args)
            completions = dict(zip(
                [x['text'] for x in result['choices']],
                [sum(x['logprobs']['token_logprobs']) for x in result['choices']]
            ))
            return completions
        except InvalidRequestError as e:
            if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
                raise PromptOverlengthError(e)
            else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def batch_top_p_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text: str = None,
                                  top_p: float = 0.9, n: int = 5, best_of: int = 10, max_new_tokens: int = 120,
                                  **kwargs) -> List[Dict[str, float]]:
        # Lazily just do them sequentially:
        if common_prompt_prefix_text:
            # Nothing fancy for OpenAI models, just concatenate the strings
            prompt_texts = [common_prompt_prefix_text + prompt_text for prompt_text in prompt_texts]

        # Lazily just do them sequentially, its not faster to batch them for API calls given rate limits
        return [
            self.top_p_lm_completion(prompt_text, top_p=top_p, n=n, best_of=best_of, max_tokens=max_new_tokens, **kwargs)
            for prompt_text in prompt_texts
        ]

    def get_completion_log_probabilities(self, prompt_text: str, completion: str, common_prompt_prefix_text: str = None,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[
        float]:
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#', ' ']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        try:
            if common_prompt_prefix_text:
                # nothing fancy for OpenAI models, just concatenate the strings
                prompt_text = common_prompt_prefix_text + prompt_text
            args = {
                "engine": self.engine,
                "prompt": prompt_text + completion,
                "max_tokens": 1,
                "logprobs": 1,
                "temperature": 0,
                "stop": stop_sequences,
                "echo": True
            }

            self.timer.step()
            result = openai.Completion.create(**args)

            print(f"got log probability for {completion}")
            tokens = result['choices'][0]['logprobs']['tokens']
            log_probs = result['choices'][0]['logprobs']['token_logprobs']
            if (prompt_text + completion) != "".join(tokens):
                # chop off last one, since we added 1 token in generation
                tokens = tokens[:-1]
            # count back to the index of the first token:
            i = len(tokens)
            remaining = completion
            while len(remaining) > 0:
                token = tokens[i - 1]
                i -= 1
                remaining = remaining[:-len(token)]

            # return the log probability of the partial sequence consisting only of the completion
            completion_token_log_probs: List[float] = log_probs[i:]
            if token_log_probs_telemetry_hook:
                token_log_probs_telemetry_hook(completion_token_log_probs)
            return completion_token_log_probs

        except InvalidRequestError as e:
            if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
                raise PromptOverlengthError(e)
            else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def batch_get_completion_log_probabilities(self, prompt_texts: List[str], completions: List[str],
                                               common_prompt_prefix_text: str = None,
                                               token_log_probs_telemetry_hook: Callable[
                                                   [List[List[float]]], None] = None) -> List[List[float]]:
        stop_sequences = self.stop_sequences or ['--', '\n', ';', '#']
        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']
        if common_prompt_prefix_text:
            # Nothing fancy for OpenAI models, just concatenate the strings
            prompt_texts = [common_prompt_prefix_text + prompt_text for prompt_text in prompt_texts]
        joint_texts = [prompt_text + completion for prompt_text, completion in zip(prompt_texts, completions)]

        # get the indices of any duplicates in joint_texts as tuples of index -> first index it duplicates
        # this will just save us tokens
        indices: List[int] = []
        de_duped_joint_texts: List[str] = []
        for i, text in enumerate(joint_texts):
            if text in de_duped_joint_texts:
                indices.append(de_duped_joint_texts.index(text))
            else:
                next_index: int = len(de_duped_joint_texts)
                de_duped_joint_texts.append(text)
                indices.append(next_index)

        try:
            args: Dict[str, Any] = {
                "engine": self.engine,
                "prompt": de_duped_joint_texts,
                "max_tokens": 1,
                "logprobs": 1,
                "temperature": 0.,
                "stop": stop_sequences,
                "organization": self.config.get('organization', openai.organization),
                "api_key": self.config.get('api_key', openai.api_key),
                "echo": True
            }

            self.timer.step()
            batch_codex_result = openai.Completion.create(**args)

            our_batch_result: List[List[float]] = []
            # in greedy mode, we can assume 1 choice per batch item
            assert len(batch_codex_result['choices']) == len(de_duped_joint_texts)
            for prompt_text, completion, de_dupe_index in zip(prompt_texts, completions, indices, strict=True):
                result = batch_codex_result['choices'][de_dupe_index]
                tokens = result['logprobs']['tokens']
                log_probs = result['logprobs']['token_logprobs']
                # count back to the index of the first token:
                i = len(tokens)
                remaining = completion
                while len(remaining) > 0:
                    token = tokens[i - 1]
                    i -= 1
                    remaining = remaining[:-len(token)]

                # return the log probability of the partial sequence consisting only of the completion
                completion_token_log_probs: List[float] = log_probs[i:]
                our_batch_result.append(completion_token_log_probs)
            if token_log_probs_telemetry_hook:
                token_log_probs_telemetry_hook(our_batch_result)
            return our_batch_result
        except InvalidRequestError as e:
            if e.user_message.startswith(TOO_MANY_TOKENS_FOR_ENGINE):
                raise PromptOverlengthError(e)
            else:
                raise e
        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def wait_until_model_ready(self, timeout_in_seconds: int = 60) -> bool:
        # Codex is always ready
        return True

