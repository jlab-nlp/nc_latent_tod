import re
from typing import List, Dict, Set, Tuple

from fuzzywuzzy import fuzz

from nc_latent_tod.acts.act import Act, Entity
from nc_latent_tod.acts.act_definitions import Request
from nc_latent_tod.delex.abstract_delexer import AbstractDelexer
from nc_latent_tod.utils.general import clean_string_for_fuzzy_comparison


SLOT_REPLACEMENTS: Dict[str, str] = {
    "ref": "reference",
    "post": "postcode",
    "addr": "address",
    "fee": "price",
    "entrancefee": "price",
    "open hours": "openhours",
    "entrance fee": "price",
    "leave at": "leave",
    "arrive by": "arrive",
    "depart": "departure",
    "dest": "destination",
    "price range": "price",
    "book people": "people",
    "book time": "time",
    "book day": "day",
    "book stay": "stay",
    "trainid": "id",  # since this will map to value_id
}


def replace_substring_fuzzy(input_string: str, query_value: str, replacement_value: str, threshold: int = 90) -> str:
    """
    Replaces an approximate match of the query found in input_string with replacement.
    If the match ratio is below the threshold, no replacement is done.
    """
    # Break the input string into words

    words: List[str] = input_string.split(' ')

    # Check fuzzy match for substrings of increasing length
    best_replacement, best_ratio = None, 0
    for start in range(len(words)):
        for num_words in range(1, len(words) + 1):
            potential_match: str = ' '.join(words[start:start+num_words])
            # when checking ratios, ignore punctuation marks
            score: int = fuzz.ratio(clean_string_for_fuzzy_comparison(potential_match), clean_string_for_fuzzy_comparison(query_value))
            if score >= threshold and score > best_ratio:
                best_replacement, best_ratio = potential_match, score
    if not best_replacement:
        return input_string
    # wrapping with these references prevents us from substituting a placeholder that is already in the string
    pattern = r"(?<!\[value_)" + re.escape(best_replacement) + r"(?!\\])"
    return re.sub(pattern, replacement_value, input_string)


def slot_name_to_placeholder(slot_name: str, transform_slots_for_eval: bool = False) -> str:
    if not transform_slots_for_eval:
        return f"[value_{slot_name}]"
    slot_name = slot_name.replace('_', ' ')
    replacement: str = f"[value_{SLOT_REPLACEMENTS.get(slot_name, slot_name)}]"
    # for whatever reason, these are not set with the value_ prefix in mwzeval. See slot name mapping here:
    # https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/cd3f0ee3a936a2d1c8567f440a0b71b215d7f991/mwzeval/normalization.py#L51-L55
    if slot_name in ('parking', 'internet', 'openhours'):
        replacement = f"[{slot_name}]"
    return replacement


def fix_placeholders(partially_delexicalized_response: str, transform_slots_for_eval: bool = False) -> str:
    """
    Given a potentially de-lexicalized response (e.g. from response gen) clean placeholders to match evaluation usage
    """
    # First, replace predicted placeholders with correct counterparts:
    # extract placeholders wrapped in brackets:
    for match in re.findall(r'\[\w+\]', partially_delexicalized_response):
        slot_name: str = re.sub(r'^value_', '', match[1:-1])
        replacement = slot_name_to_placeholder(slot_name, transform_slots_for_eval=transform_slots_for_eval)
        partially_delexicalized_response = partially_delexicalized_response.replace(match, replacement)
    return partially_delexicalized_response


class FuzzyRegexDelexer(AbstractDelexer):
    """
    A rule-based delexer using fuzzywuzzy and/or regex to delexify system responses
    """
    def delexify(self, lexicalized_response: str, system_acts: List[Act], transform_slots_for_eval: bool = False,
                 **kwargs) -> str:
        if not lexicalized_response or not system_acts:
            return lexicalized_response
        # first, fix already given placeholders (in cases where we explicitly predicted a delexicalized response)
        lexicalized_response = fix_placeholders(lexicalized_response, transform_slots_for_eval=transform_slots_for_eval)
        # Gather the slots and values present in the acts. Slots may be filled by multiple acts, with distinct values.
        slot_pairs: Set[Tuple[str, str]] = set()
        result: str = lexicalized_response
        for act in system_acts:
            # skip requests: these have no 'values' to delexicalize, only slots. We actually don't want to delexicalize
            # these, e.g. Request(service="hotel", slots=["price_range"]) -> "What is your price range for the hotel?"
            # should not be delexicalized
            if type(act) == Request:
                continue
            seen_slot_names: Set[str] = set(vars(act).keys())
            for entity in [e for e in vars(act).values() if isinstance(e, Entity)]:
                for sub_slot in vars(entity).keys():
                    seen_slot_names.add(sub_slot)
            for slot_name, value in vars(act).items():
                if isinstance(value, Entity):
                    # we'll only check one nesting level
                    for sub_slot, sub_value in vars(value).items():
                        if type(sub_value) == list:
                            for v in sub_value:
                                slot_pairs.add((sub_slot, str(v)))
                        else:
                            slot_pairs.add((sub_slot, str(sub_value)))
                elif type(value) == list:
                    # for whatever reason, we predicted multiple values. Try to delex all, but only if the value is not
                    # itself a key
                    for v in value:
                        if v not in seen_slot_names:
                            slot_pairs.add((slot_name, str(v)))
                else:
                    slot_pairs.add((slot_name, str(value)))
        # now go through each and replace with placeholders using fuzzy matching:
        for slot_name, value in sorted(slot_pairs, key=lambda x: len(x[1]), reverse=True):
            if slot_name != "service":
                replacement = slot_name_to_placeholder(slot_name, transform_slots_for_eval=transform_slots_for_eval)
                result = replace_substring_fuzzy(result, value, replacement, 90)
        return result
