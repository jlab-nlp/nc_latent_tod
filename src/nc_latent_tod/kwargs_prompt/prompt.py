import copy
import itertools
import json
import logging
import pprint
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Set, Tuple, Any, Type

from easydict import EasyDict
from nc_latent_tod.kwargs_prompt.parsing import remove_duplicate_kwargs

from nc_latent_tod.acts.act_definitions import *
from nc_latent_tod.acts.utils import convert_to_schema_slot_name
from nc_latent_tod.acts.utils import get_acts_from_system_acts, get_act_loading_context, \
    get_act_entity_classes_from_schema
from nc_latent_tod.data_types import DatasetTurn, SchemaBeliefState, ValidActsRepresentation
from nc_latent_tod.db.abstract_db import AbstractDB
from nc_latent_tod.db.types import DBEntity
from nc_latent_tod.delex.abstract_delexer import AbstractDelexer
from nc_latent_tod.delex.fuzzy_regex_delexer import FuzzyRegexDelexer
from nc_latent_tod.kwargs_prompt.templates import ABSTRACT_METHOD_TEMPLATE, INTENT_METHODS_PLACEHOLDER, \
    SERVICE_ENTITY_PLACEHOLDER, EXAMPLE_ACTS_PLACEHOLDER, ALIAS_METHOD_TEMPLATE, INFORM_CLASS_TEMPLATE, \
    PARSER_AGENT_TEMPLATE, PARSER_METHOD_TEMPLATE
from nc_latent_tod.kwargs_prompt.utils import build_docstring
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.normalization.schema_normalizer import SchemaNormalizer
from nc_latent_tod.ontology.abstract_ontology import AbstractOntology
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator, clean_slot_name, get_type_hint
from nc_latent_tod.prompting.abstract_prompt_generator import PromptMode
from nc_latent_tod.resources import read_resource
from nc_latent_tod.schemas.data_types import ServiceSchema, IntentSchema
from nc_latent_tod.schemas.utils import get_informable_slots, get_all_intents, get_intent_argument_signature
from nc_latent_tod.utils.dialogue_states import compute_delta
from nc_latent_tod.utils.dialogue_states import remove_blank_values
from nc_latent_tod.utils.general import is_jsonable, DELETE_VALUE

PROMPT_ARGUMENTS: Dict[PromptMode, List[str]] = {
    # mapping from prompt mode to ordered kwargs, with the last one being the one to predict
    "causal_dst": ["belief_state", "last_system_utterance", "user_utterance", "user_intent"],
    # p(z), completions: p(y|z)p(x|y, z)
    "noisy_channel_dst": ["belief_state", "last_system_utterance", "user_intent", "user_utterance"],
    # p(z), completions: p(y|z)p(x|y, z)
    "non_causal_sys_act_resp_only": ["system_response", "system_acts"],
    "non_causal_sys_act_resp_only_noisy_channel": ["system_acts", "system_response"],
    "causal_sys_act_policy_from_hist": ["history", "system_acts"],
    "causal_sys_act_policy_simple": ["last_system_utterance", "user_utterance", "system_acts"],
    "response_gen_simple": ["last_system_utterance", "user_utterance", "system_acts", "system_response"],
}

FORMATTING_INSTRUCTIONS: str = "Format integers as `int`s, not strings. String values do not" \
                               " need underscores for spaces. Put time values in 'hh:mm' format"


@dataclass
class ParserAgent:
    state: SchemaBeliefState


def promptify_db_result(db_result: List[DBEntity], service_name: str) -> str:
    values_str: str = ', '.join(service_name.capitalize() for _ in range(min(len(db_result), 3)))
    if len(db_result) > 3:
        values_str += ', ...'
    return '{' + f"'num_results': {len(db_result)}, 'values': [" + values_str + ']' + '}'


def prompt_item_to_str(value: Any) -> str:
    if type(value) == str:
        return json.dumps(value)
    else:
        return repr(value)


def get_in_context_example(example_number: int, arguments: OrderedDict, predict_argument: str = None,
                           already_formatted: List[str] = None):
    """
    Build a string that can be used to as an in-context example to predict the predict_argument, if given, or
    is otherwise is a complete example.

    :param example_number: example number to use in the comment
    :param arguments: arguments to pass to agent.handle_turn, in order given (as an OrderedDict)
    :param predict_argument: if given, the argument to pass to agent.handle_turn that should be predicted,
        not exemplified with a value
    :param already_formatted: if given, a list of arguments that are already formatted (e.g. as a list of ServiceAct as
        a string)
    """
    result: str = f"    # Example {example_number}"
    result += "\n    response = agent.handle_turn("
    for argument, value in arguments.items():
        result += f"\n        {argument}={prompt_item_to_str(value) if not already_formatted or not argument in already_formatted else value},"
    if predict_argument:
        result += f"\n        {predict_argument}="
    else:
        # cut off a trailing comma, typically not written even if syntactically valid
        if result.endswith(","):
            result = result[:-1]
        result += "\n    )"
    return result


def get_acts_from_belief_state_delta(delta: SchemaBeliefState) -> str:
    """
    Convert a belief state delta into a list of Inform acts, one per service.

    :param delta: belief state delta
    :return: list of Inform acts, one per service
    """
    delta = remove_blank_values(delta)
    service_acts: List[str] = []

    def repr_if_not_numeric(value: str) -> str:
        return repr(value) if not value.isnumeric() else value

    for service in delta:
        slot_pairs = {
            clean_slot_name(slot_name, service_name=service): value for slot_name, value in delta[service].items()
        }
        service_acts.append(
            f"{service.capitalize()}Inform({', '.join(slot + '=' + repr_if_not_numeric(value) for slot, value in slot_pairs.items())})"
        )

    result: str = '[' + ', '.join(service_acts) + ']'
    return result


def build_inform_class_signature(service_schema: ServiceSchema) -> str:
    """
    Given a service schema, returns a Class for Inform acts of that service

    :param service_schema: a service schema with service and slot names/types (e.g. hotel)
    :return: a method signature as a string, for use with the DialogueAgent
    """
    service_name: str = service_schema['service_name']
    argument_strings: List[str] = [
        f"{clean_slot_name(slot['name'], service_name)}: {get_type_hint(slot)}"
        for slot in get_informable_slots(service_schema)
    ]
    return INFORM_CLASS_TEMPLATE.format(
        service_class_name=service_name.capitalize(),
        service_name=service_name.lower(),
        slot_names_and_types=", ".join(argument_strings),
    )


def build_intent_method_signature(service_schema: ServiceSchema, intent: IntentSchema) -> str:
    """
    Given a service schema, returns a method signature for the intent method of that service

    :param service_schema: a service schema with service and slot names/types (e.g. hotel)
    :return: a method signature as a string, for use with the DialogueAgent
    """
    service_name: str = service_schema['service_name']
    argument_strings: List[str] = [
        f"{clean_slot_name(slot['name'], service_name)}: {get_type_hint(slot)}"
        for slot in get_informable_slots(service_schema, intents=[intent])
    ]
    method_str: str = ABSTRACT_METHOD_TEMPLATE.format(
        method_name=intent['name'],
        method_signature=", ".join(argument_strings),
        return_type="Intent",
        docstring=build_docstring(service_schema, intent=intent, informable_only=True),
    )
    return method_str


def load_dst_preamble(schema: List[ServiceSchema], combine_intents_by_signature: bool = True) -> str:
    """
    Load the initial DST task definition of the prompt as a preamble, with schema-specific user intent operations.
    """
    preamble: str = read_resource("kwargs_prompt/agent_definition.py")

    intent_method_strings: List[str] = []
    services_by_name: Dict[str, ServiceSchema] = {service['service_name']: service for service in schema}

    # first build the primary methods for each service, de-duplicating intents that have the same signature but differ
    # only in name/is_transactional. This prevents us from repeating large parts of the prompt for nearly identical
    # intents that are likely search-then-book counterparts, or similar
    all_intents: Dict[str, List[IntentSchema]] = get_all_intents(schema, allow_duplicate_signatures=True)

    # uniquely group by service x arguments to the intent
    if combine_intents_by_signature:
        intents_by_signature: Dict[str, Dict[str, List[IntentSchema]]] = defaultdict(lambda: (defaultdict(list)))
        for service_name, intents in all_intents.items():
            for intent in intents:
                intents_by_signature[service_name + get_intent_argument_signature(intent)][service_name].append(intent)
    else:
        intents_by_signature = {}
        for service_name, intents in all_intents.items():
            for intent in intents:
                signature_key = f"{service_name}_{intent['name']}"
                intents_by_signature[signature_key] = {service_name: [intent]}
    for intent_grouping in intents_by_signature.values():
        for service_name, intents in intent_grouping.items():
            # favor the non-transactional counterpart for full demonstration
            intents = sorted(intents, key=lambda i: i['is_transactional'])
            full_signature_intent: IntentSchema = intents[0]
            full_method_str: str = build_intent_method_signature(services_by_name[service_name], full_signature_intent)
            intent_method_strings.append(full_method_str)

            # for any remaining, create an alias intent method that just calls the full method
            for intent in intents[1:]:
                alias_method_str: str = ALIAS_METHOD_TEMPLATE.format(
                    method_name=intent['name'],
                    alias_name=full_signature_intent['name'],
                )
                intent_method_strings.append(alias_method_str)

    preamble = preamble.replace(INTENT_METHODS_PLACEHOLDER, "\n".join(intent_method_strings))

    # generify "nc_latent_tod.acts.act" imports.
    preamble = preamble.replace("nc_latent_tod.acts.act", "dialogue.act")
    return preamble


def load_example_acts(schema: List[ServiceSchema], loading_context: Dict[str, Any]) -> List[Act]:
    example_acts: List[Act] = []
    for act_type, service_schema in zip((Request, Offer, Inform, Affirm, ThankYou), itertools.cycle(schema)):
        if issubclass(act_type, ServiceAct):
            service_name: str = service_schema['service_name']
            service_entity_type = loading_context.get(service_name.capitalize(), None)
            # pick up to two slots with possible values
            slots = [slot for slot in service_schema['slots'] if slot['is_categorical']][:2]
            example_acts.append(
                act_type(entity=service_entity_type(**{
                    clean_slot_name(slot['name'], service_name): slot['possible_values'][0]
                    for slot in slots
                })))
        elif act_type == Request or issubclass(act_type, Request):
            service_name: str = service_schema['service_name']
            # Request(service='hotel', values=['pricerange', 'area']), etc.
            example_acts.append(
                act_type(service=service_name, values=[
                    clean_slot_name(slot['name'], service_name) for slot in service_schema['slots'][:2]
                ])
            )
        else:
            example_acts.append(act_type())

    return example_acts


def load_act_preamble_new(schema: List[ServiceSchema]) -> str:
    """
    Load the initial task definition of system act prediction as a preamble, with schema-specific Inform operations.
    In causal settings, this is a policy, in non-causal, this is dialogue act tagging.
    """
    preamble: str = read_resource("acts/act_definitions.py")

    service_entity_classes: List[str] = get_act_entity_classes_from_schema(schema)
    preamble = preamble.replace(SERVICE_ENTITY_PLACEHOLDER, "\n\n".join(service_entity_classes))
    loading_context: Dict[str, Any] = get_act_loading_context(schema)
    # generify "nc_latent_tod.*" imports.
    preamble = preamble.replace("nc_latent_tod.acts.act", "dialogue.act")
    preamble = preamble.replace("nc_latent_tod.kwargs_prompt.dialogue.management", "dialogue.management")

    # load in some example acts: request, offer, inform, affirm, thank you
    example_acts: List[Act] = load_example_acts(schema, loading_context)
    example_act_strs: List[str] = [f"{repr(act)}" for act in example_acts]
    preamble = preamble.replace(EXAMPLE_ACTS_PLACEHOLDER, "\n    ".join(example_act_strs))
    return preamble


class KwargsPromptGenerator(AbstractPromptGenerator):
    """
    A prompt generator that uses kwargs to pass arguments to the agent.handle_turn method, and uses the schema to
    generate task object definitions like user acts, system acts, etc.
    """

    preambles: Dict[str, str]
    failed_parses: List[Tuple[str, SchemaBeliefState]]
    delexer: AbstractDelexer
    normalizer: AbstractNormalizer
    act_loading_context: Dict[str, Any]

    def __init__(self, schema: List[ServiceSchema], ontology: AbstractOntology, db: AbstractDB,
                 delexer: AbstractDelexer = None, combine_intents_by_signature: bool = True) -> None:
        super().__init__(schema=schema, ontology=ontology, db=db)
        self.delexer = delexer or FuzzyRegexDelexer()
        self.preambles = {
            "dst": load_dst_preamble(schema, combine_intents_by_signature=combine_intents_by_signature),
            "act_tagging": load_act_preamble_new(schema),
        }
        self.failed_parses = []
        self.normalizer = SchemaNormalizer(schema)
        self.act_loading_context = get_act_loading_context(schema)

    def get_task_instruction(self, mode: PromptMode):
        if 'dst' in mode:
            if mode.startswith('noisy_channel'):
                return "Provide a user utterance matching the formal user_intent"
            elif 'non_causal' in mode:
                return "Provide the call(s) matching the user's intent in this context, reflected by the system response"
            else:
                return "Provide the call matching the user's intent in this context"
        elif mode.startswith('non_causal_sys_act'):
            if 'noisy_channel' in mode:
                return "Provide a system response exemplifying the given dialogue acts"
            else:
                return "Provide the dialogue acts corresponding to the observed system response"
        elif 'policy' in mode:
            if 'noisy_channel' in mode:
                return "Provide a previous turn scenario where `system_acts` makes sense as the next action"
            else:
                return "Provide dialogue acts the system should use when preparing a response to the user"
        elif 'response_gen':
            if 'noisy_channel' in mode:
                return "Respond to the user assisting with their goal and exemplifying the given dialogue acts"
            else:
                return "Provide the system response corresponding to the chosen dialogue acts"

    def get_preamble(self, prompt_mode: PromptMode) -> str:
        preamble: str
        if prompt_mode.endswith("dst"):
            preamble = self.preambles['dst']
        elif prompt_mode.startswith('non_causal_sys_act'):
            preamble = self.preambles['act_tagging']
        elif 'policy' in prompt_mode or 'response_gen' in prompt_mode:
            preamble = self.preambles['act_tagging']
        else:
            raise ValueError(f"Unexpected prompt mode: {prompt_mode}")

        # Add a task instruction + format instruction
        task_instruction = f"\n\n    # {self.get_task_instruction(prompt_mode)}"
        format_instruction: str = f"\n    # {FORMATTING_INSTRUCTIONS}"
        return preamble + task_instruction + format_instruction

    def get_completion_prefix(self, mode: PromptMode) -> str:
        if 'noisy_channel' in mode:
            return ""  # noisy channel prompts always complete with a natural language utterance from user or system
        elif mode.endswith("dst"):
            return "[agent."
        elif mode.startswith('non_causal_sys_act') or mode.startswith('causal_sys_act_policy'):
            return "["
        elif mode.startswith('response_gen'):
            # don't modify, even though we gave a leading quote, because we don't want to have to remove quotes post-hoc
            return ""
        else:
            raise ValueError(f"Unexpected prompt mode: {mode}")

    def build_in_context_example(self, turn: DatasetTurn, mode: PromptMode, example_number: int = 0,
                                 unfill_act_values: bool = False, delex_system_responses: bool = False) -> str:
        """

        """
        state_str: str = self.get_state_string(turn['last_slot_values'])
        update_intent_str: str = self.get_user_intent_str(
            prior_state=turn['last_slot_values'], next_state=turn['slot_values'],
            turn_strings=self.get_turn_strings(turn)
        )

        # use filled slot acts to delex system responses, if we asked for this
        system_response: str = turn['system_response'] or ''
        last_system_utterance: str = turn['system_utterances'][-1] if turn['system_utterances'] else None
        if delex_system_responses:
            turn_system_acts_for_delex: List[Act] = get_acts_from_system_acts(
                turn['system_response_acts'], self.schema, act_loading_context=self.act_loading_context
            )
            system_response: str = self.delexer.delexify(system_response, turn_system_acts_for_delex)
            last_system_acts_for_delex = get_acts_from_system_acts(
                turn['last_system_response_acts'], self.schema, act_loading_context=self.act_loading_context
            )
            last_system_utterance: str = self.delexer.delexify(
                turn['system_utterances'][-1], last_system_acts_for_delex
            )
        last_system_acts = get_acts_from_system_acts(turn['last_system_response_acts'], self.schema,
                                                     unfill_act_values=unfill_act_values,
                                                     act_loading_context=self.act_loading_context)
        turn_system_acts: List[Act] = get_acts_from_system_acts(turn['system_response_acts'], self.schema,
                                                                unfill_act_values=unfill_act_values,
                                                                act_loading_context=self.act_loading_context)
        supported_arguments: Dict[str, Any] = dict(
            belief_state=state_str,
            last_system_utterance=last_system_utterance,
            last_system_acts=last_system_acts,
            user_utterance=turn['user_utterances'][-1] if turn['user_utterances'] else None,
            system_response=system_response,
            system_acts=turn_system_acts,
            user_intent=update_intent_str
        )
        arguments = OrderedDict((k, supported_arguments[k]) for k in PROMPT_ARGUMENTS[mode] if k in supported_arguments)
        example: str = get_in_context_example(example_number=example_number,
                                              arguments=arguments,
                                              already_formatted=["belief_state", "user_intent", "db_result"])
        return example

    def get_dst_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                       belief_state_history: List[SchemaBeliefState], examples: List[DatasetTurn] = None,
                       last_turn_system_acts: ValidActsRepresentation = None,
                       turn_system_acts: ValidActsRepresentation = None, turn_system_response: str = None,
                       mode: PromptMode = "causal_dst") -> str:
        prompt: str = self.get_preamble(mode) + "\n\n"
        examples: List[DatasetTurn] = examples or []
        for i, example in enumerate(examples):
            # start examples at 1
            prompt += self.build_in_context_example(example, mode=mode, example_number=i + 1) + "\n\n"

        # now add the example to predict
        state_str: str = self.get_state_string(belief_state_history[-1] if belief_state_history else {})
        supported_arguments: Dict[str, Any] = dict(
            belief_state=state_str,
            last_system_utterance=turn_system_utterances[-1] if turn_system_utterances else None,
            last_system_acts=get_acts_from_system_acts(
                last_turn_system_acts, self.schema, act_loading_context=self.act_loading_context
            ) if last_turn_system_acts else [],
            user_utterance=turn_user_utterances[-1] if turn_user_utterances else None,
            system_response=turn_system_response if turn_system_response else None,
            system_acts=get_acts_from_system_acts(
                turn_system_acts, self.schema, act_loading_context=self.act_loading_context
            ) if turn_system_acts else None,
        )
        # on the 'inference' turn, we predict with everything up to user_intent. This adjustment allows for values
        # after user_intent in examples, such as user_utterance in a noisy channel prompting mode.
        turn_arguments: List[str] = PROMPT_ARGUMENTS[mode][:PROMPT_ARGUMENTS[mode].index("user_intent")]
        # further, don't set a turn argument if we don't have it supported for this input
        arguments = OrderedDict((k, supported_arguments[k]) for k in turn_arguments if k in supported_arguments)

        prompt += get_in_context_example(
            example_number=len(examples) + 1,
            arguments=arguments,
            predict_argument="user_intent",
            already_formatted=["belief_state"]
        )
        # all user intents start with this, so we can use it as a prefix
        return prompt + "[agent."

    def get_sys_act_tagging_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                                   turn_system_response: str = None,
                                   prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                                   examples: List[DatasetTurn] = None,
                                   last_turn_system_acts: ValidActsRepresentation = None,
                                   mode: PromptMode = "non_causal_sys_act_full") -> str:
        """

        """
        if not (PROMPT_ARGUMENTS[mode][-1] == "system_acts" or "noisy_channel" in mode):
            raise ValueError(f"Unexpected prompt mode={mode}: should be predicting system acts")

        prompt: str = self.get_preamble(mode) + "\n\n"
        for i, example in enumerate(examples or []):
            # start examples at 1
            prompt += self.build_in_context_example(example, mode=mode, example_number=i + 1) + "\n\n"

        # now add the example to predict
        state_str: str = self.get_state_string(prior_state or {})
        turn_strings: List[str] = turn_system_utterances[-1:] + turn_user_utterances[-1:]
        intent_str: str = self.get_act_based_user_intent_string(prior_state, next_state, turn_strings=turn_strings)
        supported_arguments: Dict[str, Any] = dict(
            belief_state=state_str,
            last_system_utterance=turn_system_utterances[-1] if turn_system_utterances else None,
            last_system_acts=get_acts_from_system_acts(
                last_turn_system_acts, self.schema, act_loading_context=self.act_loading_context
            ) if last_turn_system_acts else [],
            user_utterance=turn_user_utterances[-1] if turn_user_utterances else None,
            user_intent=intent_str,
            system_response=turn_system_response if turn_system_response else None,
        )
        if prior_state is not None and next_state is not None:
            turn_strings: List[str] = turn_system_utterances[-1:] + turn_user_utterances[-1:]
            supported_arguments['user_intent'] = self.get_act_based_user_intent_string(prior_state, next_state,
                                                                                       turn_strings)

        # on the 'inference' turn, we predict with everything up to  but excluding system_acts. This adjustment allows
        # for values after system_acts in examples, such as response in a noisy channel prompting mode.
        turn_arguments: List[str] = PROMPT_ARGUMENTS[mode][:PROMPT_ARGUMENTS[mode].index("system_acts")]
        # further, don't set a turn argument if we don't have it supported for this input
        arguments = OrderedDict((k, supported_arguments[k]) for k in turn_arguments if k in supported_arguments)

        prompt += get_in_context_example(
            example_number=len(examples) + 1 if examples else 1,
            arguments=arguments,
            predict_argument="system_acts",
            already_formatted=["belief_state", "user_intent"]
        )
        # all user intents start with this, so we can use it as a prefix
        return prompt + "["

    def get_sys_policy_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                              turn_system_utterances: List[str], turn_user_utterances: List[str],
                              prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                              examples: List[DatasetTurn] = None, mode: PromptMode = "causal_sys_act_policy",
                              db_query_service_name=None) -> str:
        if mode == "causal_sys_act_policy_from_hist" and examples:
            raise NotImplementedError("causal_sys_act_policy_from_hist for prompts with examples!")
        if not PROMPT_ARGUMENTS[mode][-1] == "system_acts":
            raise ValueError(f"Unexpected prompt mode={mode}: should be predicting system acts")
        # b_{t-1}, A_{t-1}, r_{t-1}, \Delta b_t, u_t -> A_t
        example_mode: PromptMode = "full_policy" if "noisy_channel" not in mode else mode
        prompt: str = self.get_preamble(mode) + "\n\n"
        for i, example in enumerate(examples or []):
            # start examples at 1. Always show the full turn for policy, since we want to show relationship between
            # system acts and next system response
            prompt += self.build_in_context_example(example, mode=example_mode, example_number=i + 1,
                                                    unfill_act_values=True, delex_system_responses=True) + "\n\n"

        # now add the example to predict
        state_str: str = self.get_state_string(prior_state or {})

        # these will be filled system acts still, which we can use to delex the last system utterance
        last_turn_system_acts = get_acts_from_system_acts(
            last_turn_system_acts, self.schema, act_loading_context=self.act_loading_context
        )
        turn_strings: List[str] = turn_system_utterances[-1:] + turn_user_utterances[-1:]
        intent_str: str = self.get_user_intent_str(prior_state, next_state, turn_strings=turn_strings)
        delex_last_system_utterance: str = self.delexer.delexify(turn_system_utterances[-1],
                                                                 last_turn_system_acts) if turn_system_utterances else None
        supported_arguments: Dict[str, Any] = dict(
            belief_state=state_str,
            last_system_acts=get_acts_from_system_acts(
                last_turn_system_acts, self.schema, unfill_act_values=True, act_loading_context=self.act_loading_context
            ),
            last_system_utterance=delex_last_system_utterance,
            user_utterance=turn_user_utterances[-1] if turn_user_utterances else None,
            user_intent=intent_str,
            history=pprint.pformat(self.get_history(turn_system_utterances, turn_user_utterances))
        )
        if "with_db" in mode:
            # fetch a DB result. Force the caller to supply the value of the service we want to query. If blank,
            # presume there is no relevant service to query for this turn
            if db_query_service_name:
                if db_query_service_name not in self.service_names:
                    raise ValueError(f"db_query_service_name={db_query_service_name} not in service_names, mode={mode}")
                result: List[DBEntity] = self.db.query(service_name=db_query_service_name,
                                                       constraints=next_state[db_query_service_name])
                db_string: str = promptify_db_result(result, service_name=db_query_service_name)
                supported_arguments['db_result'] = db_string
        if prior_state is not None and next_state is not None:
            turn_strings: List[str] = turn_system_utterances[-1:] + turn_user_utterances[-1:]
            supported_arguments['user_intent'] = self.get_user_intent_str(prior_state, next_state, turn_strings)

        # just puts them in the order specified in PROMPT_ARGUMENTS[mode]
        arguments = OrderedDict((k, supported_arguments[k]) for k in PROMPT_ARGUMENTS[mode] if k in supported_arguments)

        predicted_argument: str = "system_acts"
        prompt += get_in_context_example(
            example_number=len(examples) + 1 if examples else 1,
            arguments=arguments,
            predict_argument="system_acts",
            already_formatted=["belief_state", "user_intent", "db_result", "history"]
        )
        # all act lists start with this, so we can use it as a prefix
        return prompt + "["

    def get_response_gen_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                                turn_system_utterances: List[str], turn_user_utterances: List[str],
                                prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                                examples: List[DatasetTurn] = None,
                                system_response_acts: ValidActsRepresentation = None,
                                mode: PromptMode = "response_gen_simple") -> str:
        if mode == "response_gen_simple":
            policy_mode = "causal_sys_act_policy_simple"
        prompt: str = self.get_sys_policy_prompt(last_turn_system_acts=last_turn_system_acts,
                                                 turn_system_utterances=turn_system_utterances,
                                                 turn_user_utterances=turn_user_utterances, prior_state=prior_state,
                                                 next_state=next_state, examples=examples, mode=policy_mode)

        # since we lazily re-used the policy prompt, have to change its task instruction
        task_instruction = self.get_task_instruction(mode)
        policy_task_instruction = self.get_task_instruction(policy_mode)
        prompt = prompt.replace(policy_task_instruction, task_instruction)

        # now append the system response acts:
        if prompt.endswith("["):
            prompt = prompt[:-1]
        prompt += repr(get_acts_from_system_acts(
            system_response_acts, self.schema, unfill_act_values=True, act_loading_context=self.act_loading_context
        ))

        # now make room for new argument:
        prompt += ",\n        system_response=\""
        return prompt

    def parse_dst_completion(self, completion: str, state: SchemaBeliefState = None, **kwargs) -> SchemaBeliefState:
        """
        Parse LLM completion for user_acts key
        """
        try:
            # first, extract portion that is the list of acts, between brackets [ ]
            if "[" in completion and "]" not in completion:
                completion += "]"
            if "]" in completion:
                completion: str = "[" + completion[completion.index("[") + 1:completion.rindex("]")] + "]"
            parser = self.build_parser_agent(copy.deepcopy(state))
            my_locals = {'agent': parser}
            my_globals = {"dataclass": dataclass, "SchemaBeliefState": SchemaBeliefState}
            exec(completion, my_globals, my_locals)
            parser_state = my_locals['agent'].state
            new_state = copy.deepcopy(state)
            for service, slot_pairs in parser_state.items():
                for slot_name, slot_value in slot_pairs.items():
                    # if the slot_name is already in the state, then we've translated to it's original already,
                    # likely on a prior turn, so we can just copy over (if somehow its a new value, but we somehow
                    # guessed the correct original slot name, that's fine, though very unlikely). Otherwise,
                    # we look up the slot name in the original schema given the one from our prompt (e.g.
                    # 'price_range' -> 'price range'), and set the value at that original slot name instead of our
                    # made-up one that is better to prompt with.
                    orig_slot_name: str = slot_name
                    if slot_name not in state.get(service, {}):
                        orig_slot_name = self.orig_to_clean_slot_names[service].inverse[slot_name]
                    if service not in new_state:
                        new_state[service] = dict()
                    if slot_value == DELETE_VALUE:
                        new_state[service][orig_slot_name] = ""
                    else:
                        new_state[service][orig_slot_name] = str(slot_value)
            new_state = remove_blank_values(new_state)
            return new_state
        except Exception as e:
            logging.error(f"Error parsing completion: {completion}, state={pprint.pformat(state)}", e)
            self.failed_parses.append((completion, state))
            return copy.deepcopy(state)

    def remove_non_jsonables(self, act: Entity, old_completion: str = "<not_provided>"):
        for slot_name, slot_value in list(vars(act).items()):
            if isinstance(slot_value, Entity):
                self.remove_non_jsonables(slot_value)
            elif not is_jsonable(slot_value):
                logging.warning(f"Removing non-jsonable value from act: {act}, slot_name={slot_name}, "
                                f"value={slot_value}, completion={old_completion}")
                delattr(act, slot_name)

    def trim_at_end_of_list(self, completion: str) -> str:
        """
        Completions for structures (DST, Acts) are in lists surrounded by []. We should trim the completion to the list
        boundaries
        """
        num_right_brackets_needed: int = 1
        for i, c in enumerate(completion):
            if c == "[":
                # skip the first character, since we don't expect completions that start as double-lists, and we
                # probably have the completion beginning with the bracket character.
                num_right_brackets_needed += 1 if i > 0 else 0
            elif c == "]":
                num_right_brackets_needed -= 1
                if num_right_brackets_needed == 0:
                    return completion[:i + 1]
        logging.warning(f"Never trimmed completion, possibly unmatched brackets: {completion}")
        return completion

    def parse_sys_act_completion(self, completion: str, state: SchemaBeliefState = None, **kwargs) -> List[Act]:
        """
        Attempts to parse the system act predicting completion, whether from a response tagging prompt, or a policy
        prompt.

        Args:
            completion: the completion to parse
            state: the state to use to resolve references to slots in the completion, if any (optional)

        Returns:
            a list of system acts, or an empty list if the completion could not be parsed
        """
        try:
            state = state or {}  # we'll use on the off chance there are references to it, and we have them, e.g:
            # [Inform(service='hotel', area=agent.state.area)]
            # if we didn't end with the right bracket, append `]` (probably a stop-sequence)
            old_completion = completion
            if completion.count('[') > completion.count(']'):
                completion += "]"
            completion = self.trim_at_end_of_list(completion)
            assert completion.count('[') == completion.count(']'), f"Unmatched brackets: {completion}"
            assert completion.startswith("[") and completion.endswith(']'), completion
            completion = remove_duplicate_kwargs(completion)
            parser = self.build_parser_agent(copy.deepcopy(state))
            my_locals = {'agent': parser}
            my_globals = get_act_loading_context(self.schema)
            # occasionally the prompt predicts an act which was already used as an example. We'll honor this variable
            # reference.
            my_locals['act'] = load_example_acts(self.schema, loading_context=my_globals)[-1]
            result: List[Act] = eval(completion, my_globals, my_locals)

            # sometimes we get acts that aren't actually acts, e.g. [agent.no_change()], in the full prompt setting
            result = [act for act in result if isinstance(act, Act)]

            # sometimes values in the act are not strings/ints or lists of primitives, in degenerate completions
            # such as Inform(agent=agent, ...) where the value is a reference to the agent object itself
            # remove any such complex values, and log a warning:
            for act in result:
                self.remove_non_jsonables(act, old_completion=old_completion)
            return result
        except Exception as e:
            logging.error(f"Error parsing completion: {completion}, state={pprint.pformat(state)}", e)
            self.failed_parses.append((completion, state))
            return []

    def parse_response_gen_completion(self, completion: str) -> str:
        # our need to stop and produce a response string should be handled by the stop sequence, so we just return
        return completion

    def build_parser_agent(self, state: SchemaBeliefState, my_globals: Dict[str, Any] = None,
                           my_locals: Dict[str, Any] = None) -> ParserAgent:
        state = state or {}
        my_globals = my_globals or {}
        my_locals = my_locals or {}
        parser_agent_code: str = PARSER_AGENT_TEMPLATE.format(
            methods="\n".join(self.build_parser_methods(service) for service in self.schema)
        )
        parser_agent_code += "\n\nparser_agent = ParserAgent(state=state)"
        easy_state: EasyDict = EasyDict()
        for service, slot_pairs in state.items():
            easy_state[service] = EasyDict({
                # clean the slot names first, so that easy_state has a key at agent.state.hotel.price_range, instead of with a space
                clean_slot_name(slot_name, service): slot_value for slot_name, slot_value in slot_pairs.items()
            })
        my_locals = {'state': easy_state}
        my_globals = {"dataclass": dataclass, "SchemaBeliefState": SchemaBeliefState, "EasyDict": EasyDict}
        exec(parser_agent_code, my_globals, my_locals)
        return my_locals['parser_agent']

    def get_user_intent_str(self, prior_state: SchemaBeliefState, next_state: SchemaBeliefState,
                            turn_strings: List[str]) -> str:

        # first calculate and write a state string for the prior state
        delta: SchemaBeliefState = compute_delta(prior_state, next_state)
        all_intents: Dict[str, List[IntentSchema]] = get_all_intents(self.schema, allow_duplicate_signatures=False)

        def get_first_matching_intent(service_name: str, slot_pairs: Dict[str, str]) -> IntentSchema:
            """
            A belief state change could correspond to different intents, but we may not have these annotated. Instead,
            return the first intent that matches the slot pairs in the delta for that service
            """
            for intent in all_intents[service_name]:
                all_slots: Set[str] = set(intent['required_slots']).union(intent['optional_slots'].keys())
                if all(slot in all_slots for slot in slot_pairs):
                    return intent
            raise ValueError(
                f"Could not find intent for service_name={service_name}, slot_pairs={pprint.pformat(slot_pairs)}")

        # using the delta, compute which slots have been deleted/updated. We won't use a special code-path for deletions
        # and instead just set the [DELETE_VALUE]
        changes: SchemaBeliefState = defaultdict(dict)
        for service_name, slot_pairs in delta.items():
            for slot_name, slot_value in slot_pairs.items():
                changes[service_name][slot_name] = slot_value
        state_change_lines: List[str] = []
        # handle additions and updates to the state
        for service_name, slot_pairs in changes.items():
            if service_name not in self.service_names:
                logging.warning(f"Service name {service_name} not found in schema")
                continue
            intent_name: str = get_first_matching_intent(service_name, slot_pairs)['name']
            for slot_name, slot_value in slot_pairs.items():
                if not slot_value:
                    logging.error(f"blank slot value! service_name={service_name}, slot_name={slot_name}, "
                                  f"prior_state={pprint.pformat(prior_state)}, "
                                  f"gold_state={pprint.pformat(next_state)}")
                reference = self.get_state_reference(prior_state, service_name, slot_name, slot_value,
                                                     turn_strings=turn_strings)
                if reference is not None:
                    # over-write the slot value with a reference to the slot in the current state
                    referred_domain, referred_slot = reference
                    slot_pairs[
                        slot_name] = f"agent.state.{referred_domain}.{clean_slot_name(referred_slot, referred_domain)}"
            data_class_arguments: List[str] = [
                clean_slot_name(slot_name=slot_name, service_name=service_name) + '=' +
                self._quote_if_needed(service_name, slot_name, slot_value) for
                slot_name, slot_value in slot_pairs.items()
            ]
            state_change_lines.append(
                f"agent.{intent_name}({', '.join(data_class_arguments)})"
            )
        if not state_change_lines:
            state_change_lines.append("agent.no_change()")
        return "[" + ", ".join(state_change_lines) + "]"

    def get_act_based_user_intent_string(self, prior_state: SchemaBeliefState, next_state: SchemaBeliefState,
                                         turn_strings: List[str]) -> str:
        """
        To better fit the act tagging prompt, we'll construct the user's intention as Inform acts, instead of DST
        method calls.
        """
        user_acts: List[Act] = []
        delta = compute_delta(prior_state, next_state)
        # using the delta, compute which slots have been deleted/updated. We won't use a special code-path for deletions
        # and instead just set the [DELETE_VALUE]
        changes: SchemaBeliefState = defaultdict(dict)
        for service_name, slot_pairs in delta.items():
            for slot_name, slot_value in slot_pairs.items():
                changes[service_name][slot_name] = slot_value
        loading_context = get_act_loading_context(self.schema)
        for service_name, slot_pairs in changes.items():
            if service_name not in self.service_names:
                logging.warning(f"Service name {service_name} not found in schema")
                continue
            service_entity_type: Type = loading_context.get(service_name.capitalize(), Entity)
            for slot_name, slot_value in slot_pairs.items():
                if not slot_value:
                    logging.error(f"blank slot value! service_name={service_name}, slot_name={slot_name}, "
                                  f"prior_state={pprint.pformat(prior_state)}, "
                                  f"next_state={pprint.pformat(next_state)}")
                reference = self.get_state_reference(prior_state, service_name, slot_name, slot_value,
                                                     turn_strings=turn_strings)
                if reference is not None:
                    # over-write the slot value with a reference to the slot in the current state
                    referred_domain, referred_slot = reference
                    slot_pairs[
                        slot_name] = f"agent.state.{referred_domain}.{clean_slot_name(referred_slot, referred_domain)}"
            # now modify slot names to match acts
            slot_pairs = {convert_to_schema_slot_name(k, service_name=service_name): v for k, v in slot_pairs.items()}
            user_acts.append(Inform(entity=service_entity_type(**slot_pairs)))
        user_acts_intent_str: str = repr(user_acts)
        return user_acts_intent_str

    def build_parser_methods(self, service: ServiceSchema) -> str:

        service_name: str = service['service_name']
        methods: List[str] = []
        for intent in service['intents']:
            intent_args: List[str] = [
                f"{clean_slot_name(slot_name, service_name)} = None"
                for slot_name in itertools.chain(intent['required_slots'], intent['optional_slots'].keys())
            ]
            template = PARSER_METHOD_TEMPLATE
            if not intent_args:
                # this intent has no arguments, modify template to remove arg"
                template = PARSER_METHOD_TEMPLATE.replace("{intent_args}, ", "")
            methods.append(template.format(
                intent_args=", ".join(intent_args),
                intent_name=intent['name'],
                service_name=service_name,
            ))
        return "\n".join(methods)

    def get_canonical_dst_completion(self, completion: str, previous_state: SchemaBeliefState, turn_strings: List[str],
                                     mode: PromptMode) -> str:
        if not completion.startswith(self.get_completion_prefix(mode)):
            completion = self.get_completion_prefix(mode) + completion
        parse: SchemaBeliefState = self.parse_dst_completion(completion, state=previous_state)
        normal_parse = self.normalizer.normalize(parse)
        user_intent: str = self.get_user_intent_str(previous_state, normal_parse, turn_strings=turn_strings)
        return user_intent

    def get_canonical_sys_act_completion(self, completion: str, state: SchemaBeliefState = None, **kwargs):
        acts: List[Act] = self.parse_sys_act_completion(completion, state=state)
        if self.normalizer:
            acts = self.normalizer.normalize_acts(acts)
        return prompt_item_to_str(acts)

    def get_finetuning_preamble(self, prompt_mode: PromptMode) -> str:
        prompt: str = read_resource("kwargs_prompt/agent_definition.py")
        # remove blank line and placeholder markers
        prompt = prompt.replace(INTENT_METHODS_PLACEHOLDER, "").replace("\n\n\n", "\n\n")
        task_instruction = f"\n\n    # {self.get_task_instruction(prompt_mode)}"
        return prompt + task_instruction + "\n\n"

    def get_finetuning_prompt_and_completion(self, turn: DatasetTurn, mode: PromptMode, examples: List[DatasetTurn] = None) -> Tuple[str, str]:
        """
        Return a shorter prompt/completion suitable for fine-tuning, where we don't need to demonstrate intents or give
        examples
        """
        raise NotImplementedError("so far only implemented in sub-classes")

    @staticmethod
    def get_history(system_utterances: List[str], user_utterances: List[str], max_total_utterances: int = 5) -> List[str]:
        history: List[str] = []
        for sys_utt, user_utt in zip(system_utterances, user_utterances, strict=True):
            if sys_utt:
                history.append(sys_utt)
            history.append(user_utt)
        return history[-max_total_utterances:]
