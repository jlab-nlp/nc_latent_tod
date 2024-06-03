import re
from typing import List

from fuzzywuzzy import fuzz

from nc_latent_tod.data_types import SchemaBeliefState
from nc_latent_tod.utils.dialogue_states import remove_blank_values


# separate for testing
def _get_possible_values(value_str: str) -> List[str]:
    return re.split(r"[\|<>]", value_str)


# Copied from MultiWOZ 2.2 evaluation code, but in a way that's callable on a single turn.
def flatten(state_dict):
    constraints = {}
    for domain, state in state_dict.items():
        for s, v in state.items():
            constraints[(domain, s)] = v
    return constraints


def is_matching(hyp, ref, fuzzy_ratio: int = 95):
    hyp_k = hyp.keys()
    ref_k = ref.keys()
    if hyp_k != ref_k:
        return False
    for k in ref_k:
        if fuzz.partial_ratio(hyp[k], ref[k]) <= fuzzy_ratio:
            return False
    return True


def evaluate_jga(prediction: SchemaBeliefState, gold: SchemaBeliefState) -> float:
    """
    Evaluates a single prediction against a gold reference, each in standardized SchemaBeliefStateFormat format.

    :param prediction: flattened and normalized predicted slots, e.g. {"hotel-area": "centre", ...}
    :param gold: flattened and normalized gold reference slots, e.g. {"hotel-area": "centre", ...}
    :return: joint-goal accuracy
    """
    # clean both of empty values
    prediction = remove_blank_values(prediction)
    gold = remove_blank_values(gold)
    return is_matching(flatten(prediction), flatten(gold))
