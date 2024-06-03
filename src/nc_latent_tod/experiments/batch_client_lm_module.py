import abc
import copy
import logging
import time
from typing import Optional, List, Dict, Union, TypeVar, Tuple, Generic

from openai.error import RateLimitError, OpenAIError

from nc_latent_tod.acts.act import Act
from nc_latent_tod.clients.abstract_lm_client import AbstractLMClient
from nc_latent_tod.data.utils import fill_all_states
from nc_latent_tod.data_types import DatasetTurn, SchemaBeliefState
from nc_latent_tod.experiments.config import GenerationConfig
from nc_latent_tod.experiments.data_types import SchemaGuidedDSTOutputs, \
    SchemaGuidedDSTInputs, OfflineSchemaGuidedDSTInputs, GenericInputs, GenericOutputs, \
    SchemaGuidedActTaggingOutputs, SchemaGuidedActTaggingInputs, SchemaGuidedPolicyInputs, SchemaGuidedPolicyOutputs, \
    SchemaGuidedResponseGenInputs, SchemaGuidedResponseGenOutputs
from nc_latent_tod.kwargs_prompt.prompt import prompt_item_to_str
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator
from nc_latent_tod.retrieval.abstract_retriever import AbstractRetriever
from nc_latent_tod.retrieval.mpnet.contaminant_mpnet_retriever import ContaminantMPNetRetriever
from nc_latent_tod.utils.dialogue_states import clean_state
from nc_latent_tod.utils.dialogue_states import compute_delta

_ValidDSTInputType = Union[SchemaGuidedDSTInputs, OfflineSchemaGuidedDSTInputs]

# Define a TypeVar bounded by GenericInputs
TaskInput = TypeVar('TaskInput', bound=GenericInputs)

# Define a TypeVar bounded by GenericOutputs
TaskOutput = TypeVar('TaskOutput', bound=GenericOutputs)

VERBATIM_DOC_TEMPLATE = """\"\"\"
    {docs}
    \"\"\""""


class AbstractLMClientModule(Generic[TaskInput, TaskOutput], metaclass=abc.ABCMeta):
    prompt_generator: AbstractPromptGenerator
    client: AbstractLMClient
    examples: Optional[List[DatasetTurn]]
    retriever: AbstractRetriever
    retrieve_k_examples: Optional[int]
    example_warmup: int
    retriever_add_to_index: bool
    generation_cfg: GenerationConfig
    is_ready: bool = False

    def __init__(self, prompt_generator: AbstractPromptGenerator, client: AbstractLMClient,
                 examples: List[DatasetTurn] = None,
                 retriever: Optional[AbstractRetriever] = None, retrieve_k_examples: int = -1,
                 retriever_add_to_index: bool = False,
                 generation_cfg: GenerationConfig = None,
                 example_warmup: int = 0,
                 verbatim_contaminants: Optional[List[str]] = None,
                 verbatim_k_examples: int = 0,
                 **kwargs) -> None:
        """
        Build a Batch<Task> module from args:

        Attributes:
            - prompt_generator: a prompt generator which can generate prompts for the language model
            - client: a language model client which can be called with prompts and returns a completion
            - examples (optional): a list of examples to use in prompt (e.g. for in-context learning w/o a retriever)
            - retriever (optional): retriever which can be used to retrieve in-context learning examples for the prompt
            - retrieve_k_examples (optional): the number of examples to retrieve for each prompt (req: retriever)
            - retriever_add_to_index (optional): whether to add *predictions* to the retrieval index for use in later
                retrieval for prompts (req: retriever). This is a kind-of online setting useful for self-labelling, when
                 we may not want each prediction to be truly zero-shot if we have useful predictions to use as examples
        """
        self.prompt_generator = prompt_generator
        self.client = client
        self.examples = examples
        self.retriever = retriever
        self.retrieve_k_examples = retrieve_k_examples
        self.example_warmup = example_warmup
        self.retriever_add_to_index = retriever_add_to_index
        self.generation_cfg = generation_cfg or GenerationConfig()
        # allow this to be lazy initialized, and then call wait_until_ready on mode just before use
        self.is_ready = False
        self.verbatim_k_examples = verbatim_k_examples
        if verbatim_contaminants and verbatim_k_examples > 0:
            self.verbatim_retriever = ContaminantMPNetRetriever(
                documents=verbatim_contaminants,
            )

    def inject_contaminants(self, task_input: TaskInput, prompt: str) -> str:
        if self.verbatim_k_examples > 0:
            contaminated_documents = self.verbatim_retriever.get_nearest_documents(
                turn=self.build_turn_from_inputs(task_input),
                k=self.verbatim_k_examples
            )
            contaminant_string: str = VERBATIM_DOC_TEMPLATE.format(docs="\n\n    ".join(contaminated_documents))
            return prompt.replace('# Example 1', contaminant_string + "\n\n    # Example 1")

    def get_and_score_completions(self, batch_prompts: List[Tuple[str, str]], batch_inputs: List[TaskInput],
                                  batch_examples: List[List[DatasetTurn]]) -> List[Dict[str, float]]:
        if not self.is_ready:
            self.client.is_ready = self.client.wait_until_model_ready(60 * 5)  # wait up to 5 minutes
        # finally, call LLM in batch mode:
        batch_completions: List[Dict[str, float]] = None
        error_count: int = 0

        # two phases:
        # 1) get completions
        while not batch_completions:
            try:
                preamble: str = batch_prompts[0][0] if batch_prompts and batch_prompts[0] else ""
                prompt_texts: List[str] = [prompt for (_, prompt) in batch_prompts]
                batch_completions = self.client.batch_greedy_lm_completion(
                    prompt_texts=prompt_texts, common_prompt_prefix_text=preamble if preamble else None
                )
                if self.generation_cfg.generation_mode == "greedy":
                    batch_completions = self.client.batch_greedy_lm_completion(
                        prompt_texts=prompt_texts,
                        common_prompt_prefix_text=preamble if preamble else None
                    )
                elif self.generation_cfg.generation_mode.startswith("noisy_channel"):
                    batch_completions = self.client.batch_top_p_lm_completion(
                        prompt_texts=prompt_texts,
                        common_prompt_prefix_text=preamble if preamble else None,
                        **self.generation_cfg.sampling_args
                    )
                else:
                    raise ValueError(f"Unknown generation mode: {self.generation_cfg.generation_mode}")
                error_count = 0
            except (RateLimitError, OpenAIError) as e:
                time.sleep(2 ** (error_count + 1))
                error_count += 1
                if error_count > 10:
                    raise e

        # 2) re-score if needed
        if self.generation_cfg.generation_mode.startswith("noisy_channel"):
            # The standard prompt in phase 1 is in the form p(y|x, z), where z is all remaining variables in the context
            # of this turn that we conditioned on, and x is the 'input' for this particular turn (e.g. a system response
            # for an act tagger). The noisy channel prompts are in the form p(z), such that completions are in the form
            # p(y|z)p(x|y, z). Thus the log-probability of our completion is the joint probability of x and y
            # conditioned on z, but factorized in the noisy channel format. This way we only need 1 prompt per batch
            # item, since the completion will compose candidate outputs y' and the 'input' x
            new_batch_completions: List[Dict[str, float]] = []
            for candidate_completions, task_input, examples in zip(batch_completions, batch_inputs, batch_examples,
                                                                   strict=True):
                if len(candidate_completions) > 1:
                    # saving to a list to have a reliable order
                    new_completions: List[str] = list(candidate_completions.keys())
                    noisy_channel_prompts: List[Tuple[str, str]] = \
                        [self.get_noisy_channel_prompt(task_input, examples)] * len(new_completions)
                    joint_y_then_x_completions: List[Tuple[str, str, str]] = [
                        self.get_noisy_channel_completion(task_input, completion)
                        for completion in new_completions
                    ]
                    scores: List[List[float]]
                    if self.generation_cfg.generation_mode == 'noisy_channel_joint':
                        # we have p(z) in noisy_channel_prompts, and p(y|z)p(x|y, z) in joint_y_then_x_completions
                        # broken into 3 parts (y, formatting, x). We just need to join together to get a score for
                        # p(y|z)p(x|y, z)
                        completions: List[str] = ["".join(parts) for parts in joint_y_then_x_completions]
                        if len(set(prefix + prompt + completion for (prefix, prompt), completion in
                                   zip(noisy_channel_prompts, completions))) == 1:
                            # all duplicates! don't call LM to save tokens, score as log(1) = 0
                            scores = [[0.0]] * len(completions)
                        else:
                            preamble: str = noisy_channel_prompts[0][0]
                            prompt_texts: List[str] = [prompt for _, prompt in noisy_channel_prompts]
                            scores = self.client.batch_get_completion_log_probabilities(
                                prompt_texts=prompt_texts,
                                completions=completions,
                                common_prompt_prefix_text=preamble if preamble else None
                            )
                    elif self.generation_cfg.generation_mode == 'noisy_channel_cond':
                        # we have p(z) in noisy_channel_prompts, and p(y|z)p(x|y, z) in joint_y_then_x_completions
                        # broken into 3 parts (y, formatting, x). We need to add y, formatting to the prompt side
                        # and only score with completion x to get p(x|y, z)
                        preamble: str = noisy_channel_prompts[0][0]
                        prompt_texts: List[str] = [prompt + y + formatting for (_, prompt), (y, formatting, x) in
                                                   zip(noisy_channel_prompts, joint_y_then_x_completions, strict=True)]
                        completions: List[str] = [x for (_, _, x) in joint_y_then_x_completions]
                        if len(set(prompt + completion for prompt, completion in
                                   zip(prompt_texts, completions))) == 1:
                            # all duplicates! don't call LM to save tokens, score as log(1) = 0
                            scores = [[0.0]] * len(completions)
                        else:
                            scores = self.client.batch_get_completion_log_probabilities(
                                prompt_texts=prompt_texts,
                                completions=completions,
                                common_prompt_prefix_text=preamble if preamble else None
                            )
                    else:
                        raise ValueError(
                            f"Unknown noisy-channel generation mode: {self.generation_cfg.generation_mode}")
                    # re-assign scores
                    new_batch_completions.append({
                        new_completion: sum(new_scores)
                        for new_completion, new_scores in zip(new_completions, scores, strict=True)
                    })
                else:
                    # only 1 result from top-p, no need to re-score
                    new_batch_completions.append(candidate_completions)
            batch_completions = new_batch_completions
        return batch_completions

    @abc.abstractmethod
    def get_task_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> Tuple[str, str]:
        """
        Given inputs for the given turn and an optional list of examples, generate a prompt for the task.

        The prompt is the concatenation of the two returned strings, where the first string is a common prompt prefix
        for all prompts for this generator's prompt mode
        """
        pass

    @abc.abstractmethod
    def get_noisy_channel_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> Tuple[str, str]:
        """
        Given inputs for the given turn and an optional list of examples, generate a noisy channel prompt for the task.
        If the task prompt gives completions 'y' modelled by p(y|x, z), where x is the main input to the turn and z is
        all other variables conditioned on, then this gives a prompt in the form p(z), and
        get_noisy_channel_completion() gives the string for x followed by y. This way, scoring the completion with an
        LM yields p(y|z)p(x|y, z), which is the joint probability of (x, y) in noisy channel format.

        Prompt is again broken up into two strings, which can be concatenated:
         - the first string is a common prompt prefix for all prompts for this generator's prompt mode (for caching)
         - the second string is the specific prompt for this turn
        """
        pass

    @abc.abstractmethod
    def get_noisy_channel_completion(self, task_input: TaskInput, completion: str) -> Tuple[str, str, str]:
        """
        Return a formatted completion from the completion for 'y' and input 'x' from task_input whose score from an LM
        conditioned on the noisy channel prompt p(z) gives the joint probability p(y|z)p(x|y, z). We return this string
        broken into three parts, such that the full completion is from concatenating them in the 'noisy_channel_joint'
        case:

        1: y: the completion for y (e.g. the user intent string for belief state delta)
        2. formatting between y and x (e.g. keyword argument names, spacing)
        3: x: the portion of the completion for x (e.g. user utterance)
        """
        pass

    @abc.abstractmethod
    def build_turn_from_inputs(self, inputs: TaskInput) -> DatasetTurn:
        """
        From the available inputs, construct a partially mocked turn. This lets us use the same 'DTO' for all tasks,
        but prevents us from having information we shouldn't have available about a turn for a given task, provided
        we adhere to the type interface of the TaskInput
        """
        pass

    @abc.abstractmethod
    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn, completions: Dict[str, float],
                                 inputs: TaskInput, examples: List[DatasetTurn]) -> TaskOutput:
        """
        Given the prompt, the input turn, the completions, and the inputs, parse the completion(s), produce the output,
        and construct a turn that can be added to the retriever's index, if configured.
        This wraps any parsing, as well as writing output log items, etc.
        """
        # Note: could possibly break this up into multiple methods, but this works for now
        pass

    @abc.abstractmethod
    def produce_index_turn(self, input_turn: DatasetTurn, inputs: TaskInput, outputs: TaskOutput) -> DatasetTurn:
        pass

    def __call__(self, batch_inputs: List[TaskInput]) -> List[TaskOutput]:
        """
        For the given batch of inputs, run the task module and return the outputs. Where possible, runs in batch mode
        """
        batch_turns: List[DatasetTurn] = [self.build_turn_from_inputs(inputs) for inputs in batch_inputs]

        # use retriever in batch mode
        batch_examples: List[List[DatasetTurn]] = []
        if self.retriever and self.retrieve_k_examples > 0:
            if self.example_warmup > 0 and len(self.retriever) < self.example_warmup:
                logging.info(f"Retriever has {len(self.retriever)} examples, but we need {self.example_warmup}"
                             f" to warm up. Not retrieving {self.retrieve_k_examples} examples")
                batch_examples = [self.examples for _ in range(len(batch_turns))]
            else:
                # retrieve examples, and APPEND to the given examples
                batch_examples = self.retriever.get_nearest_examples_batched(
                    batch_turns,
                    k=self.retrieve_k_examples
                )
            if self.examples:
                batch_examples = [self.examples + examples for examples in batch_examples]
        else:
            batch_examples = [self.examples for _ in range(len(batch_turns))]

        # can do prompts sequentially
        batch_prompts: List[Tuple[str, str]] = []
        for examples, task_input in zip(batch_examples, batch_inputs, strict=True):
            batch_prompts.append(self.get_task_prompt(task_input=task_input, examples=examples))

        # finally, call LLM in batch mode:
        batch_completions: List[Dict[str, float]] = self.get_and_score_completions(batch_prompts,
                                                                                   batch_examples=batch_examples,
                                                                                   batch_inputs=batch_inputs)

        # do remaining parsing and processing sequentially
        results: List[TaskOutput] = []
        for (prefix, prompt), input_turn, completions, inputs, examples in zip(batch_prompts, batch_turns, batch_completions,
                                                                     batch_inputs, batch_examples, strict=True):
            outputs: TaskOutput = self.parse_and_produce_output(
                prompt=prefix + prompt, input_turn=input_turn, completions=completions, inputs=inputs, examples=examples
            )
            outputs['wandb_log_items']['batch_prompts_size'] = len(batch_prompts)
            results.append(outputs)
            if self.retriever is not None and self.retriever_add_to_index:
                # this copy is important, since fill_all_states is in-place, this modifies the dictionary
                # that was passed as input
                index_turn = self.produce_index_turn(input_turn, inputs, outputs)
                self.retriever.add_items([fill_all_states(copy.deepcopy(index_turn), inputs['schema'])])
        return results


class BatchLMClientDSTModule(AbstractLMClientModule[_ValidDSTInputType, SchemaGuidedDSTOutputs]):
    """
    A DSTModule which prompts a (possibly tuned) language model for completions that can be parsed to a belief state,
    in batched form
    """

    def __init__(self, prompt_generator: AbstractPromptGenerator, client: AbstractLMClient,
                 examples: List[DatasetTurn] = None, retriever: Optional[AbstractRetriever] = None,
                 retrieve_k_examples: int = -1, retriever_add_to_index: bool = False,
                 normalizer: AbstractNormalizer = None,
                 generation_cfg: GenerationConfig = None,
                 **kwargs) -> None:
        super().__init__(prompt_generator, client, examples, retriever, retrieve_k_examples, retriever_add_to_index,
                         generation_cfg, **kwargs)
        self.normalizer = normalizer

    def build_turn_from_inputs(self, inputs: _ValidDSTInputType) -> DatasetTurn:
        """
        Build a partially mocked DatasetTurn from available inputs

        :param inputs:
        :return:
        """
        return {
            "user_utterances": inputs['user_utterances'],
            "system_utterances": inputs['system_utterances'],
            "dialogue_id": "unknown",
            "turn_id": -1,
            "domains": [],
            "last_slot_values": inputs['belief_state_history'][-1] if len(inputs['belief_state_history']) > 0 else {},
            "turn_slot_values": {},
            "slot_values": {},
            "last_system_response_acts": inputs.get('last_system_response_acts', []),
            "system_response_acts": inputs.get('system_response_acts', []),
            "system_response": inputs.get('system_response', '')
        }

    def get_task_prompt(self, task_input: _ValidDSTInputType, examples: List[DatasetTurn]) -> Tuple[str, str]:
        """
        Calls the prompt generator to produce a prompt specific to DST
        """
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.prompt_mode)
        prompt: str = self.prompt_generator.get_dst_prompt(
            turn_user_utterances=task_input['user_utterances'],
            turn_system_utterances=task_input['system_utterances'],
            belief_state_history=task_input['belief_state_history'],
            examples=examples,
            # These might not be available in online settings
            turn_system_acts=task_input.get('system_response_acts', None),
            turn_system_response=task_input.get('system_response', None),
            mode=self.generation_cfg.prompt_mode
        )
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def get_noisy_channel_prompt(self, task_input: _ValidDSTInputType, examples: List[DatasetTurn]) -> Tuple[str, str]:
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.noisy_channel_prompt_mode)
        prompt: str = self.prompt_generator.get_dst_prompt(
            turn_user_utterances=task_input['user_utterances'],
            turn_system_utterances=task_input['system_utterances'],
            belief_state_history=task_input['belief_state_history'],
            examples=examples,
            # These might not be available in online settings
            turn_system_acts=task_input.get('system_response_acts', None),
            turn_system_response=task_input.get('system_response', None),
            mode=self.generation_cfg.noisy_channel_prompt_mode
        )
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def get_noisy_channel_completion(self, task_input: _ValidDSTInputType, completion: str) -> Tuple[str, str, str]:
        # this gets concatenated directly with the noisy channel prompt:
        previous_state: SchemaBeliefState = task_input['belief_state_history'][-1] if task_input.get(
            'belief_state_history') else {}
        turn_strings: List[str] = task_input['system_utterances'][-1:] + task_input['user_utterances'][-1:]
        completion: str = self.prompt_generator.get_canonical_dst_completion(completion, previous_state, turn_strings,
                                                                             self.generation_cfg.prompt_mode)
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion.startswith(completion_prefix):
            completion = completion[len(completion_prefix):]
        x_str: str = prompt_item_to_str(task_input['user_utterances'][-1])
        formatting: str = "],\n        user_utterance="
        if completion.endswith(
                ']'):  # I think completion includes the brackets, since we canonicalize it, but verifying
            formatting = formatting[1:]
        # joint: concatenate all. p(x|y, z) only: just x, append first two to prompt
        return completion, formatting, x_str

    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn,
                                 completions: Dict[str, float], inputs: TaskInput,
                                 examples: List[DatasetTurn]) -> TaskOutput:
        """
        Parse completion and produce outputs
        """
        # 1. choose the best completion, and prepare it for parsing
        completion: str = max(completions, key=completions.get)  # arg max
        completion = completion.strip()
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion_prefix and not completion.startswith(completion_prefix):
            completion = completion_prefix + completion
        # 2. parse the completion in the context of the current state
        current_state: SchemaBeliefState = inputs['belief_state_history'][-1] if inputs[
            'belief_state_history'] else {}
        turn_strings: List[str] = input_turn['system_utterances'][-1:] + input_turn['user_utterances'][-1:]
        cleaned_completion: str = self.prompt_generator.get_canonical_dst_completion(
            completion, current_state, turn_strings=turn_strings, mode=self.generation_cfg.prompt_mode
        )
        parse: SchemaBeliefState = self.prompt_generator.parse_dst_completion(cleaned_completion, state=current_state)

        # 3. construct and return the output blob
        output: SchemaGuidedDSTOutputs = {
            "schema_belief_state": parse,
            # infer active == updated on this turn
            "active_service_names": [service_name for service_name in input_turn['turn_slot_values']],
            "wandb_log_items": {
                "dst_retriever_pool_size": len(self.retriever) if self.retriever else 0,
                "dst_examples": len(examples) if examples else 0,
            },
            "running_log_items": {
                "completion": completion,
                "all_completions": completions,
                "raw_parse": parse,
                "dst_prompt": prompt,
                # a bit bloated but will be easier to work with instead of constantly having to look them up.
                "examples": examples or []
            }
        }
        return output

    def produce_index_turn(self, input_turn: DatasetTurn, inputs: SchemaGuidedDSTInputs,
                           outputs: SchemaGuidedDSTOutputs) -> DatasetTurn:
        # start with original turn (copy)
        predict_filled_turn: DatasetTurn = copy.deepcopy(input_turn)
        current_state: SchemaBeliefState = inputs['belief_state_history'][-1] if inputs['belief_state_history'] else {}
        parse: SchemaBeliefState = outputs['schema_belief_state']
        if self.normalizer:
            parse = self.normalizer.normalize(parse)
        predict_filled_turn['turn_slot_values'] = clean_state(compute_delta(current_state, parse),
                                                              self.prompt_generator.schema)
        predict_filled_turn['slot_values'] = clean_state(copy.deepcopy(parse), self.prompt_generator.schema)
        return predict_filled_turn


class BatchLMClientActTagModule(AbstractLMClientModule[SchemaGuidedActTaggingInputs, SchemaGuidedActTaggingOutputs]):
    """
    An act prediction module which calls an LLM with a prompt, whose completion is parsed to a list of Act
    """

    def get_task_prompt(self, task_input: SchemaGuidedActTaggingInputs, examples: List[DatasetTurn]) -> Tuple[str, str]:
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.prompt_mode)
        prompt: str = self.prompt_generator.get_sys_act_tagging_prompt(
            last_turn_system_acts=task_input.get('last_system_acts'),
            turn_system_response=task_input['system_response'],
            turn_user_utterances=task_input['user_utterances'],
            turn_system_utterances=task_input['system_utterances'],
            prior_state=task_input.get('prior_state'),
            next_state=task_input.get('next_state'),
            examples=examples,
            mode=self.generation_cfg.prompt_mode
        )
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def get_noisy_channel_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> Tuple[str, str]:
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.noisy_channel_prompt_mode)
        prompt: str = self.prompt_generator.get_sys_act_tagging_prompt(
            last_turn_system_acts=task_input.get('last_system_acts'),
            turn_system_response=task_input['system_response'],
            turn_user_utterances=task_input['user_utterances'],
            turn_system_utterances=task_input['system_utterances'],
            prior_state=task_input.get('prior_state'),
            next_state=task_input.get('next_state'),
            examples=examples,
            mode=self.generation_cfg.noisy_channel_prompt_mode
        )
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def get_noisy_channel_completion(self, task_input: TaskInput, completion: str) -> Tuple[str, str, str]:
        # first, add prefix if needed for parsing
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion_prefix and not completion.startswith(completion_prefix):
            completion = completion_prefix + completion
        # then, parse and remove it if still present
        normalized_clean_completion = self.prompt_generator.get_canonical_sys_act_completion(completion)
        if normalized_clean_completion.startswith(completion_prefix):
            normalized_clean_completion = normalized_clean_completion[len(completion_prefix):]
        # clean the completion by parsing it first
        x_str: str = prompt_item_to_str(task_input['system_response'])
        # join(y, formatting, x) => joint_y_then_x completion
        return normalized_clean_completion, ",\n        system_response=", x_str

    def build_turn_from_inputs(self, inputs: SchemaGuidedActTaggingInputs) -> DatasetTurn:
        turn_slot_values: SchemaBeliefState = compute_delta(inputs.get('prior_state', {}), inputs.get('next_state', {}))
        return {
            "user_utterances": inputs['user_utterances'],
            "system_utterances": inputs['system_utterances'],
            "dialogue_id": "unknown",
            "turn_id": -1,
            "domains": [],
            "last_slot_values": inputs.get('prior_state', {}),
            "turn_slot_values": turn_slot_values,
            "slot_values": inputs.get('next_state', {}),
            "last_system_response_acts": inputs.get('last_system_acts'),
            "system_response_acts": [],  # we'll write to this in the end
            "system_response": inputs.get('system_response', '')
        }

    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn,
                                 completions: Dict[str, float], inputs: SchemaGuidedActTaggingInputs,
                                 examples: List[DatasetTurn]) -> SchemaGuidedActTaggingOutputs:
        # 1. choose the best completion, and prepare it for parsing
        completion: str = max(completions, key=completions.get)  # arg max
        completion = completion.strip()
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion_prefix and not completion.startswith(completion_prefix):
            completion = completion_prefix + completion

        normalized_completion = self.prompt_generator.get_canonical_sys_act_completion(completion)

        # 2. parse the completion
        parse: List[Act] = self.prompt_generator.parse_sys_act_completion(normalized_completion,
                                                                          state=inputs.get('next_state', {}))
        json_friendly_parse: List[str] = [act.to_json() for act in parse]

        # 3. construct and return the output blob
        output: SchemaGuidedActTaggingOutputs = {
            "system_response_acts": json_friendly_parse,
            "active_service_names": self.prompt_generator.get_service_names_from_acts(parse),
            "wandb_log_items": {
                "act_tag_retriever_pool_size": len(self.retriever) if self.retriever else 0,
                "act_tag_examples": len(examples) if examples else 0,
            },
            "running_log_items": {
                "completion": normalized_completion,
                "all_completions": completions,
                "raw_parse": json_friendly_parse,
                "act_tag_prompt": prompt,
                # a bit bloated but will be easier to work with instead of constantly having to look them up.
                "examples": examples or []
            }
        }
        return output

    def produce_index_turn(self, input_turn: DatasetTurn, inputs: SchemaGuidedActTaggingInputs,
                           outputs: SchemaGuidedActTaggingOutputs) -> DatasetTurn:
        prev_acts = inputs.get('last_system_acts', [])
        json_friendly_prev_acts = [act.to_json() for act in prev_acts]
        predict_filled_turn: DatasetTurn = copy.deepcopy(input_turn)
        predict_filled_turn['last_system_response_acts'] = json_friendly_prev_acts
        predict_filled_turn['system_response_acts'] = outputs['system_response_acts']
        return predict_filled_turn


class BatchLMClientPolicyModule(AbstractLMClientModule[SchemaGuidedPolicyInputs, SchemaGuidedPolicyOutputs]):
    """
    An act prediction module which calls an LLM with a prompt, whose completion is parsed to a list of Act
    """

    def get_task_prompt(self, task_input: SchemaGuidedPolicyInputs, examples: List[DatasetTurn]) -> Tuple[str, str]:
        delta: SchemaBeliefState = compute_delta(task_input.get('prior_state', {}), task_input.get('next_state', {}))
        service_name: str = list(delta.keys())[0] if delta else None
        if len(delta) > 1:
            logging.warning(f"More than one service in delta for DB query: {delta}, picking {service_name} arbitrarily")
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.prompt_mode)
        prompt: str = self.prompt_generator.get_sys_policy_prompt(
            last_turn_system_acts=task_input.get('last_system_acts'),
            turn_system_utterances=task_input['system_utterances'],
            turn_user_utterances=task_input['user_utterances'],
            prior_state=task_input.get('prior_state'),
            next_state=task_input.get('next_state'), examples=examples,
            mode=self.generation_cfg.prompt_mode,
            db_query_service_name=service_name)
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def build_turn_from_inputs(self, inputs: SchemaGuidedPolicyInputs) -> DatasetTurn:
        return {
            "user_utterances": inputs['user_utterances'],
            "system_utterances": inputs['system_utterances'],
            "dialogue_id": "unknown",
            "turn_id": -1,
            "domains": [],
            "last_slot_values": {},
            "turn_slot_values": {},
            "slot_values": {},
            "last_system_response_acts": inputs.get('last_system_acts'),
            "system_response_acts": [],  # we'll write to this in the end
            "system_response": ""  # we don't get to know this
        }

    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn,
                                 completions: Dict[str, float], inputs: SchemaGuidedPolicyInputs,
                                 examples: List[DatasetTurn]) -> SchemaGuidedPolicyOutputs:
        # 1. choose the best completion, and prepare it for parsing
        completion: str = max(completions, key=completions.get)  # arg max
        completion = completion.strip()
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion_prefix and not completion.startswith(completion_prefix):
            completion = completion_prefix + completion

        # 2. parse the completion
        parse: List[Act] = self.prompt_generator.parse_sys_act_completion(completion,
                                                                          state=inputs.get('next_state', {}))
        json_friendly_parse: List[str] = [act.to_json() for act in parse]

        # 3. construct the output blob
        output: SchemaGuidedPolicyOutputs = {
            "system_response_acts": json_friendly_parse,
            "active_service_names": self.prompt_generator.get_service_names_from_acts(parse),
            "wandb_log_items": {
                "policy_retriever_pool_size": len(self.retriever) if self.retriever else 0,
                "policy_examples": len(examples) if examples else 0,
            },
            "running_log_items": {
                "policy_completion": completion,
                "all_policy_completions": completions,
                "raw_parse": json_friendly_parse,
                "policy_prompt": prompt,
                "examples": examples or []
            }
        }

        # 4. write back the predictions to the input turn, so that it might be added to the retrieval index
        predict_filled_turn: DatasetTurn = input_turn
        predict_filled_turn['system_response_acts'] = json_friendly_parse
        return output

    def produce_index_turn(self, input_turn: DatasetTurn, inputs: SchemaGuidedPolicyInputs,
                           outputs: SchemaGuidedPolicyOutputs) -> DatasetTurn:
        predict_filled_turn: DatasetTurn = copy.deepcopy(input_turn)
        predict_filled_turn['system_response_acts'] = outputs['system_response_acts']
        return predict_filled_turn

    def get_noisy_channel_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> str:
        raise NotImplementedError()

    def get_noisy_channel_completion(self, task_input: TaskInput, completion: str) -> str:
        raise NotImplementedError()


class BatchLMClientResponseGenModule(
    AbstractLMClientModule[SchemaGuidedResponseGenInputs, SchemaGuidedResponseGenOutputs]
):
    """
    A response generation module
    """

    def get_task_prompt(self, task_input: SchemaGuidedResponseGenInputs, examples: List[DatasetTurn]) -> \
            Tuple[str, str]:
        preamble: str = self.prompt_generator.get_preamble(self.generation_cfg.prompt_mode)
        prompt: str = self.prompt_generator.get_response_gen_prompt(
            last_turn_system_acts=task_input.get('last_system_acts'),
            turn_user_utterances=task_input['user_utterances'],
            turn_system_utterances=task_input['system_utterances'],
            prior_state=task_input.get('prior_state'),
            next_state=task_input.get('next_state'),
            examples=examples,
            system_response_acts=task_input['system_response_acts'],
            mode=self.generation_cfg.prompt_mode
        )
        if self.verbatim_k_examples > 0:
            prompt = self.inject_contaminants(task_input, prompt)
        assert prompt.startswith(preamble), "Prompt does not start with preamble"
        return preamble, prompt[len(preamble):]

    def build_turn_from_inputs(self, inputs: SchemaGuidedResponseGenInputs) -> DatasetTurn:
        return {
            "user_utterances": inputs['user_utterances'],
            "system_utterances": inputs['system_utterances'],
            "dialogue_id": "unknown",
            "turn_id": -1,
            "domains": [],
            "last_slot_values": {},
            "turn_slot_values": {},
            "slot_values": {},
            "last_system_response_acts": inputs.get('last_system_acts'),
            "system_response_acts": inputs['system_response_acts'],
            "system_response": ""  # we don't get to know this
        }

    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn,
                                 completions: Dict[str, float],
                                 inputs: SchemaGuidedResponseGenInputs,
                                 examples: List[DatasetTurn]) -> SchemaGuidedResponseGenOutputs:
        # 1. choose the best completion, and prepare it for parsing
        completion: str = max(completions, key=completions.get)  # arg max
        completion = completion.strip()
        completion_prefix: str = self.prompt_generator.get_completion_prefix(self.generation_cfg.prompt_mode)
        if completion_prefix and not completion.startswith(completion_prefix):
            completion = completion_prefix + completion

        # 2. parse the completion
        parsed_response: str = self.prompt_generator.parse_response_gen_completion(completion)

        # 3. construct and return the output blob
        output: SchemaGuidedResponseGenOutputs = {
            "system_response": parsed_response,
            "wandb_log_items": {
                "rg_retriever_pool_size": len(self.retriever) if self.retriever else 0,
                "rg_examples": len(examples) if examples else 0,
            },
            "running_log_items": {
                "rg_completion": completion,
                "all_rg_completions": completions,
                "raw_parse": parsed_response,
                "rg_prompt": prompt,
                "examples": examples or []
            }
        }
        return output

    def produce_index_turn(self, input_turn: DatasetTurn, inputs: SchemaGuidedResponseGenInputs,
                           outputs: SchemaGuidedResponseGenOutputs) -> DatasetTurn:
        predict_filled_turn: DatasetTurn = copy.deepcopy(input_turn)
        predict_filled_turn['system_response'] = outputs['system_response']
        return predict_filled_turn

    def get_noisy_channel_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> str:
        raise NotImplementedError()

    def get_noisy_channel_completion(self, task_input: TaskInput, completion: str) -> str:
        raise NotImplementedError()
