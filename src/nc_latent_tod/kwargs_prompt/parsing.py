import re
from typing import List, Dict, Tuple, Set


def remove_duplicate_kwargs(completion):
    fixed_class_strings: List[str] = []
    for string in split_on_classes(completion):
        # see if it has an argument that is itself a class definition:
        # entity=Hotel(name='hotel california', ...), etc.
        # entity_matches: Dict[str, str] = {k: '' for k in re.findall(r'\b\w+=\w+\([^()]*\)', string)}
        entity_matches: Dict[str, str] = {k: '' for k in re.findall(r'(?<!^)\b(?:\w+=)?\w+\([^()]*\)(?!$)', string)}
        # For now, replace each entity match a placeholder!
        for i, match in enumerate(entity_matches):
            # added some random bits just to be sure, and structured as a kwarg so it is captured
            placeholder: str = f"__ENTITY_PLACEHOLDER_{i}=3111152526557254315__"
            string = string.replace(match, placeholder)
            entity_matches[match] = placeholder

        # Now operate on string without entity match: Find all keyword arguments
        matches: List[Tuple, Tuple] = match_keyword_arguments(string)

        seen: Set = set()
        unique_matches: List[Tuple[str, str]] = []

        # Add to unique_matches only if keyword not seen before
        for slot, value in matches:
            if slot not in seen:
                seen.add(slot)
                unique_matches.append((slot, value))

        # Reconstruct the string
        args = ', '.join(f"{k}={v}" for k, v in unique_matches)
        fixed_string: str = re.sub(r'\(.*\)', f'({args})', string)

        # now replace placeholders with recursively de-duplicated values
        for match, placeholder in entity_matches.items():
            fixed_string = fixed_string.replace(placeholder, remove_duplicate_kwargs(match))
        fixed_class_strings.append(fixed_string)
    result: str = "".join(fixed_class_strings)
    return result


def split_on_classes(completion: str) -> List[str]:
    count = 0
    start = 0
    result = []

    for i, char in enumerate(completion):
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count == 0:
                result.append(completion[start:i + 1].strip())
                start = i + 1
    if start < len(completion):
        result.append(completion[start:].strip())

    # Filter out empty strings (like those caused by commas or newlines between class instances).
    return [x for x in result if x]


def match_keyword_arguments(string: str) -> List[Tuple[str, str]]:
    matches = []
    # all keyword arguments of the form key='value' (strings)
    matches.extend(re.findall(r'(\w+)=(\'[^\']*\')', string))
    # all in the form key=value (ints, variables, etc. Must not be a class, hence the negative lookahead)
    var_matches = re.findall(r'(\w+)=([\w\.]+)[^\w\(]', string)
    matches.extend(var_matches)
    # all in the form key=[value1, value2, ...] (lists)
    matches.extend(re.findall(r'(\w+)=(\[.*?\])', string))
    return matches
