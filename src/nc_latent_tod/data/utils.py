from typing import List, Optional

from datasets import Features, Value, Sequence, load_dataset, DatasetDict

from nc_latent_tod.data_types import SchemaBeliefState, DatasetTurn, ValidActsRepresentation
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.schemas.utils import get_all_informable_slot_names


def get_hf_dataset_features(schema: List[ServiceSchema]) -> Features:
    belief_state_feature = {}
    for service_name, slot_names in get_all_informable_slot_names(schema).items():
        belief_state_feature[service_name] = {slot_name: Value("string") for slot_name in slot_names}
    return Features({
        "dialogue_id": Value(dtype="string"),
        "turn_id": Value(dtype="int8"),
        "domains": Sequence(Value("string")),
        "user_utterances": Sequence(Value("string")),
        "system_utterances": Sequence(Value("string")),
        "slot_values": belief_state_feature,
        "turn_slot_values": belief_state_feature,
        "last_slot_values": belief_state_feature,
        "last_system_response_acts": Sequence(Value(dtype="string")),
        "system_response_acts": Sequence(Value(dtype="string")),
        "system_response": Value(dtype="string")
    })


def get_hf_empty_belief_state(schema: List[ServiceSchema]) -> SchemaBeliefState:
    """
    Huggingface datasets do not tolerate missing keys, so for each possible service, slot, and value, we need a key
    present with missing values represented as empty strings. This produces a full state structure of exclusively empty 
    string values, for the given schema
    
    :param schema: schema to construct empty state for 
    :return: empty state with services and slots populated by empty strings
    """
    state: SchemaBeliefState = {}
    for service_name, slot_names in get_all_informable_slot_names(schema).items():
        state[service_name] = {}
        for slot_name in slot_names:
            state[service_name][slot_name] = ""
    return state


def fill_hf_empty_state(state: SchemaBeliefState, schema: List[ServiceSchema]) -> SchemaBeliefState:
    """                                                                       
    Huggingface datasets do not tolerate missing keys, so for each possible service, slot, and value, we need a key
    present with missing values represented as empty strings. This produces a full state structure of exclusively empty 
    string values, for the given schema
    
    :param schema: schema to construct empty state for 
    :return: empty state with services and slots populated by empty strings
    """
    for service_name, slot_names in get_all_informable_slot_names(schema).items():
        state[service_name] = state.get(service_name, {})
        for slot_name in slot_names:
            state[service_name][slot_name] = state[service_name].get(slot_name, "")
    return state


def fill_hf_empty_acts(acts: ValidActsRepresentation) -> ValidActsRepresentation:
    if not acts:
        return [""]
    if isinstance(acts, list):
        return [act if type(act) == str else act.to_json() for act in acts]
    elif isinstance(acts, dict):
        return {
            "act": acts.get("act", []),
            "service": acts.get("service", []),
            "slot_name": acts.get("slot_name", []),
            "value": acts.get("value", [])
        }
    else:
        raise ValueError(f"acts must be a list of Act/strings or a dict of lists of strings, but got {type(acts)}")


def fill_all_states(turn: DatasetTurn, schema: List[ServiceSchema]) -> DatasetTurn:
    turn['slot_values'] = fill_hf_empty_state(turn['slot_values'], schema)
    turn['last_slot_values'] = fill_hf_empty_state(turn['last_slot_values'], schema)
    turn['turn_slot_values'] = fill_hf_empty_state(turn['turn_slot_values'], schema)
    turn['last_system_response_acts'] = fill_hf_empty_acts(turn.get('last_system_response_acts', {}))
    turn['system_response_acts'] = fill_hf_empty_acts(turn.get('system_response_acts', {}))
    turn['system_response'] = turn.get('system_response', "")
    return turn


def load_possibly_missing_dataset(dataset_name_or_path: str) -> Optional[DatasetDict]:
    try:
        return load_dataset(dataset_name_or_path)
    except FileNotFoundError as e:
        return None


if __name__ == '__main__':
    features = get_hf_dataset_features(read_multiwoz_schema())
    print(features)
