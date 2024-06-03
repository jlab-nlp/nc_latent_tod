import copy
import re
from collections import defaultdict
from typing import List, Dict

import dictdiffer
from nc_latent_tod.schemas.utils import get_all_informable_slot_names

from nc_latent_tod.schemas.data_types import ServiceSchema

from nc_latent_tod.data_types import SchemaBeliefState
from nc_latent_tod.utils.general import DELETE_VALUE


def remove_blank_values(state: SchemaBeliefState) -> SchemaBeliefState:
    new_state: SchemaBeliefState = {}
    for service in state:
        if not state[service]:
            continue
        for slot, value in state[service].items():
            if value:
                if service not in new_state:
                    new_state[service] = {}
                new_state[service][slot] = value
    return new_state


def apply_update(state: SchemaBeliefState, update: SchemaBeliefState) -> SchemaBeliefState:
    state = copy.deepcopy(state)
    for service in update:
        state[service] = state.get(service, {})
        for slot, value in update[service].items():
            if value == DELETE_VALUE:
                del state[service][slot]
            else:
                state[service][slot] = value
    state = remove_blank_values(state)
    return state


def get_state_slots_as_key_string(state: SchemaBeliefState) -> str:
    """
    Returns a string which uniquely identifies this state by its non-empty slots
    """
    return str(sorted([f"{service}.{slot}" for service, slots in state.items() for slot in slots if slots[slot]]))


def clean_state(state: SchemaBeliefState, schema: List[ServiceSchema]) -> SchemaBeliefState:
    cleaned_state = {}
    informable_slots: Dict[str, List[str]] = get_all_informable_slot_names(schema)
    for service_name, slot_pairs in state.items():
        if service_name in informable_slots:
            cleaned_state[service_name] = {}
            for slot_name, value in slot_pairs.items():
                if slot_name in informable_slots[service_name] and value is not None:
                    cleaned_state[service_name][slot_name] = value
    return cleaned_state


def compute_delta(before: SchemaBeliefState, after: SchemaBeliefState) -> SchemaBeliefState:
    """
    Compute the difference between two dialogue states, each in flattened form

    :param before: previous dialogue state (e.g. prior turn)
    :param after: current dialogue state (e.g. this turn)
    :return: difference, as its own dialogue state
    """
    delta: SchemaBeliefState = defaultdict(dict)
    new_diffs = []
    for diff in dictdiffer.diff(remove_blank_values(before), remove_blank_values(after)):
        # initial step: treat changes as adds, since we're starting from an empty dict (UPSERT like behavior)
        if diff[0] == 'change':
            service_name, slot_name = diff[1].split('.')
            new_diffs.append(('add', service_name, [(slot_name, diff[2][1])]))
        elif diff[0] == 'remove':
            # treat removals as adding the special DELETE_VALUE in the location of the deleted key
            if not diff[1]:
                # we removed the last key in these service(s), add individual add-diffs per slot
                for service_name, slot_pairs in diff[2]:
                    new_diffs.append(('add', service_name, [(k, DELETE_VALUE) for k in slot_pairs]))
            else:
                new_diffs.append(('add', diff[1], [(slot_name, DELETE_VALUE) for slot_name, _ in diff[2]]))
        elif diff[0] == 'add':
            new_diffs.append(diff)
    for diff in new_diffs:
        delta = dictdiffer.patch([diff], delta)
    return dict(delta)


def get_icdst_slot_name(slot_name: str) -> str:
    """
    Given the slot_name ('price range') form the IC-DST slot name ('pricerange') by applying any normalization rules

    :param slot_name: name of the slot
    :return: slot-only name in IC-DST format (ignores service prefix, use get_icdst_full_slot_name to include)
    """
    slot_name: str = slot_name.replace(' ', '')
    return re.sub(r'book(\S)', lambda match: f'book {match.group(1)}', slot_name)


def get_mwzeval_db_slot_name(slot_name: str) -> str:
    """
    Given the slot_name ('price range') return the version of this slot name that could be looked up in
    Tomiinek/MultiWOZ_Evaluation's MultiWOZVenueDatabase
    """
    slot_name = get_icdst_slot_name(slot_name)
    return slot_name.replace('arriveby', 'arrive').replace('leaveat', 'leave')


def get_icdst_full_slot_name(service_name: str, slot_name: str) -> str:
    """
    Given the service_name ('hotel') and slot_name ('price range') form the IC-DST slot name ('hotel-pricerange')
    :param service_name: name of the service
    :param slot_name: name of the slot
    :return: combined full slot name in IC-DST format
    """
    return f"{service_name}-{get_icdst_slot_name(slot_name)}"


if __name__ == '__main__':
    before: SchemaBeliefState = {
        "hotel": {"name": "test", "type": "hotel", "parking": "yes"},
        "taxi": {"destination": "all saints"},
        "train": {"destination": "all saints"},
    }
    after: SchemaBeliefState = {
        "hotel": {"name": "test", "type": "guest house", "area": "west"},
        "attraction": {"area": "west"}
    }
    print(compute_delta(before, after))