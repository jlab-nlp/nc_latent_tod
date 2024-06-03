import copy
from typing import List, Dict, Set, Optional, Iterable, Callable, Any

import wordninja
from fuzzywuzzy import fuzz

from nc_latent_tod.acts.act import Act
from nc_latent_tod.acts.utils import get_acts_from_system_acts
from nc_latent_tod.data_types import SchemaBeliefState, ValidActsRepresentation
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.schemas.data_types import ServiceSchema, SlotSchema
from nc_latent_tod.utils.general import clean_string_for_fuzzy_comparison

SCHEMA_AGNOSTIC_VALID_KEYS: Set[str] = {
    # for all acts
    'entity',
    '__type',
    # For NotifyFailure
    'reason',
    # for Inform
    'choice',
    'id',
    'num_choices',
    # for Request
    'service',
    'values',
}


def is_fuzzy_match(predicted_value: str, schema_value: str, ratio: int = 90) -> bool:
    """
    Checks if the predicted value matches the schema value, using fuzzy matching.
    """
    observed_ratio: int = fuzz.ratio(
        clean_string_for_fuzzy_comparison(predicted_value),
        clean_string_for_fuzzy_comparison(schema_value)
    )
    return observed_ratio >= ratio


def get_fuzzy_match(predicted_value: str, schema_values: Iterable[str], ratio: int = 90) -> Optional[str]:
    """
    Checks if the predicted value matches the schema value, using fuzzy matching.
    """
    matches: List[str] = [
        schema_value for schema_value in schema_values
        if is_fuzzy_match(predicted_value, schema_value, ratio)
    ]
    assert len(matches) <= 1, f"Multiple matches found for {predicted_value}: {', '.join(matches)}"
    return matches[0] if matches else None


def build_valid_act_keys(schema: List[ServiceSchema]) -> Set[str]:
    valid_act_keys = copy.deepcopy(SCHEMA_AGNOSTIC_VALID_KEYS)
    # these are the keys coming directly from the schema.
    for service in schema:
        for slot in service['slots']:
            valid_act_keys.add(slot['name'])
            # also accept slot name with underscores
            valid_act_keys.add(slot['name'].replace(' ', '_'))
            # break up compound words with spaces
            split_slot_name: str = ' '.join(wordninja.split(slot['name']))
            valid_act_keys.add(split_slot_name)
            # also accept slot name with underscores
            valid_act_keys.add(split_slot_name.replace(' ', '_'))
    return valid_act_keys


class SchemaNormalizer(AbstractNormalizer):
    schema: List[ServiceSchema]
    slot_schemas: Dict[str, Dict[str, SlotSchema]]
    valid_act_keys: Set[str]

    def __init__(self, schema: List[ServiceSchema]):
        self.schema = schema
        self.slot_schemas = {}
        for service in schema:
            self.slot_schemas[service['service_name']] = {slot['name']: slot for slot in service['slots']}
        self.valid_act_keys = build_valid_act_keys(schema)
        self.valid_service_names: List[str] = [service['service_name'] for service in schema]
        self.valid_slot_names: Dict[str, Set[str]] = {
            service['service_name']: set(slot['name'] for slot in service['slots']) for service in schema
        }

    def normalize(self, raw_parse: SchemaBeliefState) -> SchemaBeliefState:
        """
        This normalizer simply tries to infer matching `possible_values` for slots which specify the complete
        set of possible values, removing predictions which cannot be mapped to a slot through fuzzy matching.
        """
        new_parse: SchemaBeliefState = {}
        for service_name in raw_parse:
            matching_service_name: Optional[str] = get_fuzzy_match(service_name, self.valid_service_names)
            if not matching_service_name:
                continue
            new_parse[matching_service_name] = {}
            for slot_name in raw_parse[service_name]:
                matching_slot_name: Optional[str] = \
                    get_fuzzy_match(slot_name, self.valid_slot_names[matching_service_name])
                if not matching_slot_name:
                    continue
                slot_schema: SlotSchema = self.slot_schemas[matching_service_name][matching_slot_name]
                possible_values: List[str] = slot_schema.get('possible_values', [])
                if slot_schema.get('is_categorical'):
                    # we have possible values to check, skip if the predicted value is not in the set
                    matching_value: Optional[str] = get_fuzzy_match(raw_parse[service_name][slot_name], possible_values)
                    if matching_value:
                        new_parse[matching_service_name][matching_slot_name] = matching_value
                else:
                    new_parse[matching_service_name][matching_slot_name] = raw_parse[service_name][slot_name]
        return new_parse

    def normalize_acts(self, acts: ValidActsRepresentation,
                       telemetry_hook_for_removed_slot_pairs: Callable[[str, Any], Any] = None) -> List[Act]:
        acts: List[Act] = get_acts_from_system_acts(copy.deepcopy(acts), self.schema)
        for act in acts:
            # get all slot names present in the act:
            for key, value in act.to_dict().items():
                if key == 'entity':
                    for k, v in value.items():
                        if k not in self.valid_act_keys:
                            if telemetry_hook_for_removed_slot_pairs:
                                # for now, don't distinguish between act slots and entity slots
                                telemetry_hook_for_removed_slot_pairs(k, v)
                            delattr(act.entity, k)
                if key not in self.valid_act_keys:
                    if telemetry_hook_for_removed_slot_pairs:
                        telemetry_hook_for_removed_slot_pairs(key, value)
                    delattr(act, key)
        return acts
