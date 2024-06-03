import json
import logging
import pprint
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

from nc_latent_tod.acts.utils import get_acts_from_system_acts
from nc_latent_tod.data_types import SchemaBeliefState, ValidActsRepresentation, DatasetTurn
from nc_latent_tod.kwargs_prompt.prompt import KwargsPromptGenerator, promptify_db_result, PROMPT_ARGUMENTS, \
    get_in_context_example
from nc_latent_tod.prompting.abstract_prompt_generator import PromptMode
from nc_latent_tod.utils.dialogue_states import compute_delta

SPLIT_STRING: str = "# Example 1"


EMPTY_TURN: DatasetTurn = {
    'dialogue_id': "empty",
    'turn_id': -1,
    'domains': [],
    'system_utterances': [],
    'user_utterances': [],
    'slot_values': {},
    'turn_slot_values': {},
    'last_slot_values': {},
    'last_system_response_acts': [],
    'system_response_acts': [],
    'system_response': "",
}


class SimpleFTKwargsPromptGenerator(KwargsPromptGenerator):

    def get_preamble(self, mode: PromptMode) -> str:
        """
        Return a prefix common to all prompts with this mode, if any. An empty string is returned if there is no known
        common prefix for the mode.
        """
        return super().get_finetuning_preamble(mode)

    def get_dst_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                       belief_state_history: List[SchemaBeliefState], examples: List[DatasetTurn] = None,
                       last_turn_system_acts: ValidActsRepresentation = None,
                       turn_system_acts: ValidActsRepresentation = None, turn_system_response: str = None,
                       mode: PromptMode = "causal_dst", **kwargs) -> str:
        turn = {
            **EMPTY_TURN,
            'user_utterances': turn_user_utterances,
            'system_utterances': turn_system_utterances,
            # Turn 0 could be empty list/None
            'last_slot_values': belief_state_history[-1] if belief_state_history else {},
            'slot_values': {},  # unknown!
        }
        # return only the prompt (_ = completion)
        ft_prompt, _ = self.get_finetuning_prompt_and_completion(turn, mode=mode, examples=examples)
        if 'noisy_channel' in mode:
            ft_prompt = ft_prompt[:ft_prompt.rindex("user_intent=[agent.")] + "user_intent=[agent."
        return ft_prompt

    def get_sys_act_tagging_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                                   turn_system_response: str = None, prior_state: SchemaBeliefState = None,
                                   next_state: SchemaBeliefState = None, examples: List[DatasetTurn] = None,
                                   last_turn_system_acts: ValidActsRepresentation = None,
                                   turn_system_acts: ValidActsRepresentation = None,
                                   mode: PromptMode = "non_causal_sys_act_resp_only") -> str:
        if turn_system_acts:
            assert 'noisy_channel' in mode, "don't pass turn_system_acts in non-noisy channel mode"
        turn = {
            **EMPTY_TURN,
            'user_utterances': turn_user_utterances,
            'system_utterances': turn_system_utterances,
            'system_response': turn_system_response,
            'last_slot_values': prior_state,
            'slot_values': next_state,
            'last_system_response_acts': last_turn_system_acts,
        }
        # return only the prompt (_ = completion)
        ft_prompt, _ = self.get_finetuning_prompt_and_completion(turn, mode=mode, examples=examples)
        # noisy channel prompt will have dangling key for response from the call above, trim it since we add it back
        # in the act tagging module:
        if 'noisy_channel' in mode:
            ft_prompt = ft_prompt[:ft_prompt.rindex("system_acts=[")] + "system_acts=["
        return ft_prompt

    def get_sys_policy_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                              turn_system_utterances: List[str], turn_user_utterances: List[str],
                              prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                              examples: List[DatasetTurn] = None, mode: PromptMode = "causal_sys_act_policy",
                              db_query_service_name: str = None) -> str:
        turn = {
            **EMPTY_TURN,
            'last_system_response_acts': last_turn_system_acts,
            'system_utterances': turn_system_utterances,
            'user_utterances': turn_user_utterances,
            'last_slot_values': prior_state,
            'slot_values': next_state,
        }
        # return only the prompt (_ = completion)
        ft_prompt, _ = self.get_finetuning_prompt_and_completion(turn, mode=mode, examples=examples)
        return ft_prompt

    def get_response_gen_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                                turn_system_utterances: List[str], turn_user_utterances: List[str],
                                prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                                examples: List[DatasetTurn] = None,
                                system_response_acts: ValidActsRepresentation = None,
                                mode: PromptMode = "response_gen_simple") -> str:
        turn = {
            **EMPTY_TURN,
            'last_system_response_acts': last_turn_system_acts,
            'system_utterances': turn_system_utterances,
            'user_utterances': turn_user_utterances,
            'last_slot_values': prior_state,
            'slot_values': next_state,
            'system_response_acts': system_response_acts,
        }
        # return only the prompt (_ = completion)
        ft_prompt, _ = self.get_finetuning_prompt_and_completion(turn, mode=mode, examples=examples)
        return ft_prompt + "\""  # adding here and not as a prefix to avoid post-hoc quote removal. Trusting LM output.

    def get_finetuning_prompt_and_completion(self, turn: DatasetTurn, mode: PromptMode, examples: List[DatasetTurn] = None) -> Tuple[str, str]:
        prompt = super().get_finetuning_preamble(mode)

        if mode == "causal_sys_act_policy_from_hist" and examples:
            logging.error(f"causal_sys_act_policy_from_hist mode does not support examples, "
                          f"but {len(examples)} were passed. Ignoring.")
            examples = None

        # ====================== Setting flags for argument formatting ======================
        unfill_act_values: bool = "policy" in mode or 'response_gen' in mode
        delex_system_responses: bool = "policy" in mode or 'response_gen' in mode

        # ====================== Computing arguments accordingly ======================
        last_system_acts: ValidActsRepresentation = turn['last_system_response_acts']
        if last_system_acts:
            last_system_acts = get_acts_from_system_acts(
                acts=last_system_acts,
                schema=self.schema,
                unfill_act_values=unfill_act_values,
                act_loading_context=self.act_loading_context
            )
        turn_system_acts: ValidActsRepresentation = turn['system_response_acts']
        if turn_system_acts:
            turn_system_acts = get_acts_from_system_acts(
                acts=turn_system_acts,
                schema=self.schema,
                unfill_act_values=unfill_act_values,
                act_loading_context=self.act_loading_context
            )
        state_str: str = self.get_state_string(turn['last_slot_values'])
        intent_str: str = self.get_user_intent_str(turn['last_slot_values'], turn['slot_values'],
                                                   turn_strings=self.get_turn_strings(turn))
        delta = compute_delta(turn['last_slot_values'], turn['slot_values'])

        system_response: str = turn['system_response']
        last_system_utterance: str = turn['system_utterances'][-1]
        if delex_system_responses:
            last_acts_for_delex = get_acts_from_system_acts(
                turn['last_system_response_acts'], schema=self.schema, act_loading_context=self.act_loading_context
            ) if last_system_acts else []
            last_system_utterance = self.delexer.delexify(last_system_utterance, last_acts_for_delex)
            turn_acts_for_delex = get_acts_from_system_acts(
                turn['system_response_acts'], schema=self.schema, act_loading_context=self.act_loading_context
            ) if turn_system_acts else []
            system_response = self.delexer.delexify(system_response, turn_acts_for_delex)
        supported_arguments: Dict[str, Any] = dict(
            belief_state=state_str,
            user_intent=intent_str,
            last_system_utterance=last_system_utterance,
            last_system_acts=last_system_acts,
            user_utterance=turn['user_utterances'][-1],
            system_response=system_response,
            system_acts=turn_system_acts,
            history=pprint.pformat(self.get_history(turn['system_utterances'], turn['user_utterances']))
        )
        example_i: int = 1
        # policy modes use unfilled act values (placeholders), e.g. Inform(hotel=Hotel(name='[value_name]'))

        if examples:
            example_mode: PromptMode = mode if 'policy' not in mode else "full_policy"
            for example in examples:
                prompt += self.build_in_context_example(
                    turn=example,
                    mode=example_mode,
                    example_number=example_i,
                    unfill_act_values=unfill_act_values,
                    delex_system_responses=delex_system_responses,
                ) + "\n\n"
                example_i += 1
        if "with_db" in mode:
            db_query_service_name: str = list(delta.keys())[0] if delta else None
            db_result = db_query_service_name and promptify_db_result(
                self.db.query(service_name=db_query_service_name,
                              constraints=turn['slot_values'][db_query_service_name]),
                service_name=db_query_service_name) or None
            supported_arguments['db_result'] = db_result
        # on the 'inference' turn, we predict with everything up to user_intent. This adjustment allows for values
        # after user_intent in examples, such as user_utterance in a noisy channel prompting mode.
        turn_arguments: List[str] = PROMPT_ARGUMENTS[mode][:-1]
        # further, don't set a turn argument if we don't have it supported for this input
        arguments = OrderedDict((k, supported_arguments[k]) for k in turn_arguments if k in supported_arguments)

        predict_argument: str = PROMPT_ARGUMENTS[mode][-1]
        prompt += get_in_context_example(
            example_number=example_i,
            arguments=arguments,
            predict_argument=predict_argument,
            already_formatted=["belief_state", "user_intent", "db_result", "history"],
        )
        prefix = self.get_completion_prefix(mode)
        prompt += prefix
        completion = supported_arguments[predict_argument]
        if predict_argument not in ["belief_state", "user_intent", "db_result"]:
            completion = repr(completion) if type(completion) != str else json.dumps(completion)
        assert completion.startswith(prefix), "expected completion to start with prefix"
        completion = completion[len(prefix):]
        return prompt, completion
