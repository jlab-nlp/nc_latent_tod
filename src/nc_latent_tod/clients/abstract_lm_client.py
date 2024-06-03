import abc
from typing import List, Dict, Callable


class AbstractLMClient(metaclass=abc.ABCMeta):
    """
    Any 'client' implementing these methods should be able to support a RefPyDST experiment as the generative LM
    """

    @abc.abstractmethod
    def __init__(self, stop_sequences: List[str] = None, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def greedy_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for LM Completion
        :param common_prompt_prefix_text: prefix text for LM Completion. If specified, the true prompt to the LM
            will be the concatenation of common_prompt_prefix_text and prompt_text. Some clients may use this to cache
            the prefix computation of past_key_values for long prompts with shared prefixes.
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        pass

    @abc.abstractmethod
    def top_p_lm_completion(self, prompt_text: str, common_prompt_prefix_text: str = None, top_p: float = 0.9,
                            n: int = 5, best_of: int = 10, max_new_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """
        Given a prompt, generate a completion using the given engine and other completion parameters.

        :param prompt_text: prefix text for LM Completion
        :param common_prompt_prefix_text: prefix text for LM Completion. If specified, the true prompt to the LM
            will be the concatenation of common_prompt_prefix_text and prompt_text. Some clients may use this to cache
            the prefix computation of past_key_values for long prompts with shared prefixes.
        :return: the single most likely completion for the prompt (greedily sampled), not including the prompt tokens.
        """
        pass

    @abc.abstractmethod
    def batch_top_p_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text: str = None,
                                  top_p: float = 0.9, n: int = 5, best_of: int = 10, max_new_tokens: int = 120,
                                  **kwargs) -> List[Dict[str, float]]:
        pass

    @abc.abstractmethod
    def get_completion_log_probabilities(self, prompt_text: str, completion: str, common_prompt_prefix_text: str = None,
                                         token_log_probs_telemetry_hook: Callable[[List[float]], None] = None) -> List[
        float]:
        """
        :param prompt_text: prefix text for LM Completion
        :param completion: the completion to score
        :param common_prompt_prefix_text: prefix text for LM Completion. If specified, the true prompt to the LM
            will be the concatenation of common_prompt_prefix_text and prompt_text. Some clients may use this to cache
            the prefix computation of past_key_values for long prompts with shared prefixes.
        :param token_log_probs_telemetry_hook: a call-back function that takes a list of log probabilities for each
            token in the completion, called once with output (i.e. for visualizations, logging, etc.).
        """
        pass

    @abc.abstractmethod
    def batch_greedy_lm_completion(self, prompt_texts: List[str], common_prompt_prefix_text: str = None,
                                   max_new_tokens: int = 128) -> List[Dict[str, float]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def batch_get_completion_log_probabilities(self, prompt_texts: List[str], completions: List[str],
                                               common_prompt_prefix_text: str = None,
                                               token_log_probs_telemetry_hook: Callable[
                                                   [List[List[float]]], None] = None) -> List[List[float]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_until_model_ready(self, timeout_in_seconds: int = 60) -> bool:
        """
        For API based models, particularly in self-hosted configs, this waits at most timeout_in_seconds seconds
        until the model is ready for use. Return True once model is ready, or False if it never becomes ready
        """
        pass
