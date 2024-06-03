import abc
import logging
import re
from collections import defaultdict
from typing import List, Collection, Tuple, Dict, Optional, Literal, Set

import wordninja
from bidict import bidict
from fuzzysearch import find_near_matches
from nc_latent_tod.db.abstract_db import AbstractDB

from nc_latent_tod.acts.act import Act
from nc_latent_tod.data_types import SchemaBeliefState, DatasetTurn, ValidActsRepresentation
from nc_latent_tod.ontology.abstract_ontology import AbstractOntology
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.data_types import SlotSchema

PromptMode = Literal[
    "causal_dst",  # b_{t-1}, r_{t-1}, u_t -> b_t
    "noisy_channel_dst",  # b_{t-1}, r_{t-1} -> b_t -> u_t
    "non_causal_sys_act_resp_only",  # r_t -> A_t
    "non_causal_sys_act_resp_only_noisy_channel",  # A_t -> r_t
    "causal_sys_act_policy_from_hist",  # ..., u_{t-1}, r_{t-1}, u_t -> A_t
    "causal_sys_act_policy_simple",  # r_{t-1}, u_t, -> A_t
    "response_gen_simple"  # r_{t-1}, u_t, A_t -> r_t
]


def _index_slot_schemas(schema: List[ServiceSchema]) -> Dict[str, Dict[str, SlotSchema]]:
    slot_schemas: Dict[str, Dict[str, SlotSchema]] = defaultdict(dict)
    for service in schema:
        service_slots: Dict[str, SlotSchema] = {slot['name']: slot for slot in service['slots']}
        slot_schemas[service['service_name']] = {
            clean_slot_name(slot_name=k, service_name=service['service_name']): v
            for k, v in service_slots.items()
        }
    return slot_schemas


class AbstractPromptGenerator(metaclass=abc.ABCMeta):
    """
    A base class for prompt generators with some useful utilities. Assumes that the generator at least needs to be
    aware of the schema, and contains some common python-prompting utility methods.
    """

    schema: List[ServiceSchema]
    ontology: AbstractOntology
    slot_schemas: Dict[str, Dict[str, SlotSchema]]
    orig_to_clean_slot_names: Dict[str, bidict[str, str]]
    completion_prefix: str = ""
    service_names: Set[str]
    db: AbstractDB

    def __init__(self, ontology: AbstractOntology, schema: List[ServiceSchema], db: AbstractDB) -> None:
        super().__init__()
        self.schema = schema
        self.ontology = ontology
        self.slot_schemas = _index_slot_schemas(self.schema)
        self.orig_to_clean_slot_names = self._index_slot_names(self.schema)
        self.service_names = set(service['service_name'] for service in schema)
        self.db = db

    def get_slot_schema(self, service_name: str, slot_name: str) -> SlotSchema:
        """
        Return the SlotSchema for the given service and slot name
        """
        try:
            return self.slot_schemas[service_name][clean_slot_name(slot_name=slot_name, service_name=service_name)]
        except KeyError as e:
            raise ValueError(f"slot schema {service_name}-{slot_name} not found", e)

    def _quote_if_needed(self, service_name: str, slot_name: str, slot_value: str):
        if slot_value.startswith("agent.state."):
            # reference, don't quote
            return slot_value
        type_hint: str = get_type_hint(self.get_slot_schema(service_name, slot_name))
        if (type_hint.startswith("int") or type_hint.startswith("float")) and slot_value and slot_value.isnumeric():
            return slot_value
        return f'"{slot_value}"'

    @staticmethod
    def clean_utterance(utterance: str) -> str:
        # trying to keep this very minimal, but some is needed to ensure a valid python prompt
        text = utterance
        # removing line breaks
        text = re.sub(r'[\r\n]+', '', text)
        return text

    def get_service_state_string(self, service_name: str, slot_pairs: Collection[Tuple[str, str]],
                                 use_dataclass: bool = True):
        """
        Return a string for showing the current belief state for a particular service

        :param service_name: name of the service
        :param slot_pairs: slot value pairs that are currently populated
        :param use_dataclass: whether to represent the state service values as their own dataclasses, or as dictionaries
        :return:
        """
        arguments: List[str] = [
            f"{clean_slot_name(slot_name=slot_name, service_name=service_name)}=" +
            f"{self._quote_if_needed(service_name, slot_name, slot_value)}"
            for slot_name, slot_value in slot_pairs if slot_value
        ]
        return f"{service_name.capitalize() if use_dataclass else 'dict'}({', '.join(arguments)})"

    def get_state_string(self, prior_state: SchemaBeliefState, delexicalize_non_categoricals: bool = False,
                         use_dataclass: bool = False) -> str:
        """
        Return a string representing a line for the prior state, as could be included in a prompt

        :param prior_state: dialogue prior state to represent
        :param delexicalize_non_categoricals: if True, replace each non-categorical value with a pseudo "state-reference"
        :return: string representing the state, on a single line
        """
        # we'll want to look up SlotSchemas by service and slot names, so pre-compute
        service_names: List[str] = sorted(list(prior_state.keys()))
        service_state_arguments: List[str] = []
        for service_name in service_names:
            # many slot values might be blank - we should ignore these
            if any(slot_value for slot_value in prior_state[service_name].values()):
                service_state_arguments.append(
                    f"{service_name}={self.get_service_state_string(service_name, prior_state[service_name].items(), use_dataclass=use_dataclass)}"
                )
        return f"BeliefState({', '.join(service_state_arguments)})"

    @staticmethod
    def has_fuzzy_match(slot_value: str, turn_strings: List[str]) -> bool:
        """
        Return true if the slot value is present as a fuzzy matched string within one or more of the turn strings.
        We use a fuzzy search with a maximum Levenstein distance of 2 for slot values of 6 or more characters, 1 for
        slot values of 5 characters, and zero otherwise. Maximum deletions is always 1. Capitalization is fully
        ignored.

        :param slot_value: slot value to query for (e.g. wednesday)
        :param turn_strings: turn strings which may mention this value in a fuzzy surface form
            (e.g. ['i need a hotel starting on wendesday'])
        :return: whether a match exists
        """
        max_l_dist: int = 2
        if len(slot_value) < 5:
            max_l_dist = 0
        if len(slot_value) < 6:
            max_l_dist = 1
        for turn_string in turn_strings:
            if not turn_string:
                logging.warning(f"passed an empty turn string: [{', '.join(turn_strings)}], slot_value={slot_value}")
            elif len(find_near_matches(slot_value.lower(), turn_string.lower(), max_l_dist=max_l_dist,
                                       max_deletions=1)) > 0:
                return True
        return False

    def get_state_reference(self, prior_state_norm: SchemaBeliefState, service_name: str, slot_name: str,
                            slot_value: str, turn_strings: List[str]) -> Optional[Tuple[str, str]]:
        # dontcare is not a co-referable dialogue item in practice (a user would not really say I feel about the
        # price-range the same way I do about the hotel-type as an indication of "dontcare")
        if slot_value == "dontcare":
            return None

        if self.ontology.is_numeric(service_name, slot_name) or self.ontology.is_bool(service_name, slot_name):
            # this is a number or a boolean. For this data-type, meaning is dependent on the slot name (i.e. I would
            # refer to the same number of people, not "5" in abstract. I could, but would be unlikely to say "book me
            # hotel for the same number of nights as people in my restaurant reservation")
            #
            # Luckily, for all integer and time based slots, MultiWOZ uses the same slot_name for the same kind of
            # meaning. This makes programming this rule easier, but it should still be generally doable in other
            # ontologies, since we are making this determination based two gold state dictionaries, not an un-labelled
            # turn.
            for state_domain, state_pairs in prior_state_norm.items():
                # for a slot-pair that is numeric/bool to be considered as co-referring to something in the state:
                # 1) the slot value must exactly match the slot value for some other slot in the state
                # 2) the number/bool must NOT be in either of the turn strings
                # 3) the slot-name (excluding domain) must match
                # i.e. we choose to favor coincidental numeral equivalence as NOT co-reference, i.e. (hotel-stay, 6) is
                # unlikely to co-refer to (restaurant-book people, 6), even if we can't find an occurrence of 6 in the
                # turn strings: 6 nights != 6 people
                if slot_name in state_pairs and not any(f" {slot_value} " in s for s in turn_strings) and \
                        slot_value == state_pairs[slot_name]:
                    return state_domain, slot_name
            # it was a numeric, but there was no occurrence in the state
            return None

        # next check for explicit mentions of the slot value. For numerics, these can be false positive if there is overlap
        # on value, hence the prior check. For names, days of the week, etc, this is less likely
        if self.has_fuzzy_match(slot_value, turn_strings=turn_strings):
            # this slot-value is explicitly mentioned in the turn. This will misinterpret statements like the following:
            # state = {restaurant-book_people = 2} "i need a hotel for the same group of people for 2 nights "
            return None

        # finally, check if the slot is mentioned in the state. Since we're comparing two normalized states, ideally we
        # shouldn't need to do any fuzzy string matching (still, we'll ignore casing)
        for state_domain, state_pairs in prior_state_norm.items():
            for state_slot_name, state_value in state_pairs.items():
                if type(state_value) == str and type(slot_value) == str and slot_value.lower() == state_value.lower():
                    return state_domain, state_slot_name
        return None

    @staticmethod
    def get_turn_strings(turn: DatasetTurn) -> List[str]:
        turn_strings = []
        if turn['system_utterances'] and turn['system_utterances'][-1]:
            turn_strings.append(turn['system_utterances'][-1])
        turn_strings.append(turn['user_utterances'][-1])
        return turn_strings

    def _index_slot_names(self, schema: List[ServiceSchema]) -> Dict[str, bidict[str, str]]:
        result = {}
        for service in schema:
            service_slot_mapping = {}
            for slot in service['slots']:
                orig_slot_name = slot['name']
                cleaned_slot_name: str = clean_slot_name(orig_slot_name, service['service_name'])
                # this way, when we produce a normalized belief state we aren't labelling the service on two portions,
                # i.e. {'hotel': {'hotel-parking': 'yes}}. There is probably a cleaner approach than this, as there are
                # multiple translation points across the system
                short_orig_slot_name = orig_slot_name.replace(f"{service['service_name']}-", "")
                service_slot_mapping[short_orig_slot_name] = cleaned_slot_name
            result[service['service_name']] = bidict(service_slot_mapping)
        return result

    @abc.abstractmethod
    def get_dst_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                       belief_state_history: List[SchemaBeliefState], examples: List[DatasetTurn] = None,
                       **kwargs) -> str:
        """
        Given DST inputs, return a prompt for the DST module to complete

        Args:
            turn_user_utterances: the user utterances for the turn (ending in current user utterance)
            turn_system_utterances: the system utterances for the turn (ending in system utterance prior to most recent user)
            belief_state_history: previous belief state predictions, such that [-1] holds the previous belief state
            examples: the examples to use for the prompt
        """
        pass

    @abc.abstractmethod
    def parse_dst_completion(self, completion: str, state: SchemaBeliefState, **kwargs) -> SchemaBeliefState:
        """
        Given a completion and prior state, return a new state dictionary with the completion applied

        Args:
            completion: the completion string
            state: the prior state (SchemaBeliefState)

        Returns:
            the new state (SchemaBeliefState)
        """
        pass

    @abc.abstractmethod
    def get_sys_act_tagging_prompt(self, *, turn_user_utterances: List[str], turn_system_utterances: List[str],
                                   turn_system_response: str = None,
                                   prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                                   examples: List[DatasetTurn] = None,
                                   last_turn_system_acts: ValidActsRepresentation = None,
                                   mode: PromptMode = "non_causal_sys_act_full") -> str:
        """
        Given act tagging inputs, return a prompt for the act tagging module to complete

        Args:
            turn_user_utterances: the user utterances for the turn (ending in current user utterance)
            turn_system_utterances: the system utterances for the turn (ending in system utterance prior to most recent user)
            turn_system_response: the system response for the turn (ending in current system response: non-causal!)
            prior_state: the prior state (SchemaBeliefState)
            next_state: the next state (SchemaBeliefState) as a result of the user's utterance (still causal)
            examples: the examples to use for the prompt
            last_turn_system_acts: the system acts from the last turn (pairs with turn_system_utterances[-1])
            mode: the mode of the prompt, which determines which inputs are used and how they are organized

        Returns:
            the prompt string
        """
        pass

    @abc.abstractmethod
    def get_sys_policy_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                              turn_system_utterances: List[str], turn_user_utterances: List[str],
                              prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                              examples: List[DatasetTurn] = None,
                              mode: PromptMode = "causal_sys_act_policy",
                              db_query_service_name: str = None) -> str:
        # A_{t-1}, r_{t-1}, u_t, b_{t-1}, \Delta b_t -> A_t
        """
        Given policy inputs, return a prompt for the policy module to complete (produces acts, which can be parsed same
        as act tagging)

        Args:
            last_turn_system_acts: the system acts from the last turn (pairs with turn_system_utterances[-1])
            turn_user_utterances: the user utterances for the turn (ending in current user utterance)
            turn_system_utterances: the system utterances for the turn (ending in system utterance prior to most recent user)
            prior_state: the prior state (SchemaBeliefState)
            next_state: the next state (SchemaBeliefState) as a result of the user's utterance (still causal)
            examples: the examples to use for the prompt
            mode: the mode of the prompt, which determines which inputs are used and how they are organized
            db_query_service_name: the service name to query the DB for, if any

        Returns:
            the prompt string
        """
        pass

    @abc.abstractmethod
    def get_response_gen_prompt(self, *, last_turn_system_acts: ValidActsRepresentation = None,
                                turn_system_utterances: List[str], turn_user_utterances: List[str],
                                prior_state: SchemaBeliefState = None, next_state: SchemaBeliefState = None,
                                examples: List[DatasetTurn] = None,
                                system_response_acts: ValidActsRepresentation = None,
                                mode: PromptMode = "response_gen_simple") -> str:
        # b_{t-1}, A_{t-1}, r_{t-1}, b_t, u_t, A_t -> r_t
        """
        Given response generation inputs, return a prompt for the response generation module to complete

        Args:
            last_turn_system_acts: the system acts from the last turn (pairs with turn_system_utterances[-1])
            turn_user_utterances: the user utterances for the turn (ending in current user utterance)
            turn_system_utterances: the system utterances for the turn (ending in system utterance prior to most recent user)
            prior_state: the prior state (SchemaBeliefState)
            next_state: the next state (SchemaBeliefState) as a result of the user's utterance (still causal)
            examples: the examples to use for the prompt
            mode: the mode of the prompt, which determines which inputs are used and how they are organized
            system_response_acts: the acts which the response should embody (e.g. from a policy)
        """
        pass

    @abc.abstractmethod
    def parse_sys_act_completion(self, completion: str, state: SchemaBeliefState = None,
                                 **kwargs) -> List[Act]:
        """
        Attempts to parse the system act predicting completion, whether from a response tagging prompt, or a policy
        prompt.

        Args:
            completion: the completion to parse
            state: the state to use to resolve references to slots in the completion, if any (optional)

        Returns:
            a list of system acts, or an empty list if the completion could not be parsed
        """
        pass

    @abc.abstractmethod
    def get_completion_prefix(self, mode: PromptMode) -> str:
        pass

    @abc.abstractmethod
    def parse_response_gen_completion(self, completion: str) -> str:
        pass

    @abc.abstractmethod
    def get_canonical_dst_completion(self, completion: str, previous_state: SchemaBeliefState, turn_strings: List[str], mode: PromptMode) -> str:
        pass

    @abc.abstractmethod
    def get_canonical_sys_act_completion(self, completion: str, state: SchemaBeliefState = None, **kwargs) -> str:
        pass

    def get_service_names_from_acts(self, acts: List[Act]) -> List[str]:
        service_names: List[str] = []
        for act in acts:
            if hasattr(act, 'entity'):
                possible_service_name: str = type(act.entity).__name__.lower()
                if possible_service_name in self.service_names:
                    service_names.append(possible_service_name)
        return service_names

    def get_finetuning_prompt_and_completion(self, turn: DatasetTurn, mode: PromptMode, examples: List[DatasetTurn] = None) -> Tuple[str, str]:
        pass

    def get_preamble(self, mode: PromptMode) -> str:
        """
        Return a prefix common to all prompts with this mode, if any. An empty string is returned if there is no known
        common prefix for the mode.
        """
        # assume we don't have one, sub-classes may add one
        return ""


def clean_slot_name(slot_name: str, service_name: str) -> str:
    """
    Clean slot names by:
    - remove service name from slot name
    - lowercase
    - replace '-' with '_'
    - remove leading/trailing non [a-z] characters
    :param slot_name: slot name to clean
    :param service_name: name of the service
    :return:
    """
    slot_name = slot_name.lower()
    slot_name = slot_name.replace('-', '_')
    slot_name = slot_name.replace(service_name, '')
    slot_name = "_".join(wordninja.split(slot_name))
    slot_name = re.sub(r'^[^a-z]+', '', slot_name)
    slot_name = re.sub(r'[^a-z]+$', '', slot_name)
    return slot_name


def get_example_values(slot_schema: SlotSchema, quote_values: bool = True) -> str:
    if not slot_schema.get('possible_values') or not slot_schema['is_categorical']:
        return ""
    delimeter = "', '" if quote_values else ", "
    possible_values_str: str = delimeter.join(
        value for value in slot_schema['possible_values'][:4])
    if quote_values:
        possible_values_str = "'" + possible_values_str + "'"
    if len(slot_schema['possible_values']) > 4:
        possible_values_str = "e.g. " + possible_values_str + ", ..."
    return possible_values_str


def get_type_hint(slot_schema: SlotSchema, include_values_comment: bool = False, dontcare_can_be_numeric: bool = False,
                  include_default_value: bool = True) -> str:
    if slot_schema['is_categorical']:
        considered_values: List[str] = slot_schema['possible_values']
        if not dontcare_can_be_numeric:
            considered_values = [value for value in considered_values if value != "dontcare"]
        # check for numeric types, starting with positive/negative integers
        if all(value.strip().isdigit() or (value.startswith("-") and value[1:].isdigit())
               for value in considered_values):
            return "int = None" if include_default_value else "int"
        elif all(value.strip().isnumeric() for value in considered_values):
            # we'll call all other numerics floats
            return "float = None" if include_default_value else "float"
        # Else, categorical slots give their values. We could define custom types (e.g. Literals) for each, but
        # for now we'll just suffix with a comment providing some possible values (first four)
        if include_values_comment:
            possible_values_str = get_example_values(slot_schema)
            type_hint = f"str {'= None' if include_default_value else ''}  # {possible_values_str}"
        else:
            type_hint = "str = None" if include_default_value else "str"
        return type_hint
    else:
        # non-categorical slots will be strings
        return "str = None" if include_default_value else "str"
