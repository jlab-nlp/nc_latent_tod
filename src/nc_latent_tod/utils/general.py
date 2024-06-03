import json
import logging
import os
import string
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Tuple, TypeVar, List, Dict, Union, Optional, Type

from nc_latent_tod.data_types import RawMultiWOZDialogue

NC_LATENT_TOD_OUTPUTS_DIR: str = "NC_LATENT_TOD_OUTPUTS_DIR"
DELETE_VALUE: str = "[DELETE]"

def get_project_output_root_dir() -> str:
    output_dir: Optional[str] =  os.environ.get(NC_LATENT_TOD_OUTPUTS_DIR)
    if not output_dir:
        logging.warning(f"Environment variable {NC_LATENT_TOD_OUTPUTS_DIR} not set. Using current directory: {os.getcwd()}/outputs")
        return f"{os.getcwd()}/outputs"
    return output_dir

def read_json(json_path: str) -> Any:
    with open(json_path, "r") as f:
        return json.load(f)


class FunctionsAsNamesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.FunctionType):
            return obj.__name__
        return super(FunctionsAsNamesEncoder, self).default(obj)


def write_json(data: Any, path: str, indent: int = 0, cls: Type[json.JSONEncoder] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        return json.dump(data, f, indent=indent, cls=cls)


def read_lines(file_path: str, strip: bool=True) -> List[str]:
    with open(file_path, "r") as f:
        return [l.strip() if strip else l for l in f.readlines()]


def check_argument(assertion: Any, message: Optional[str]) -> None:
    """
    Checks the assertion is true, and raises a Value error if it is false, with the provided message
    :param assertion: assertion to check (can be a truthy, such as a non-empty vs. empty list)
    :param message: message to provide if assertion fails
    :return: None
    """
    if not assertion:
        raise ValueError(message)


def get_output_dir_full_path(file_path: Union[str, Path]) -> Union[str, Path]:
    """
    If the file_path is not absolute, return the file path such that it is rooted in the configured outputs directory

    :param file_path: relative or absolute path. relative paths will be re-mapped to relative to REFPYDST_OUTPUT_DIR
      or outputs/ by default. Absolute paths are preserved.
    :return: file path with any modifications
    """
    if not os.path.isabs(file_path) and NC_LATENT_TOD_OUTPUTS_DIR in os.environ:
        return os.path.join(os.environ[NC_LATENT_TOD_OUTPUTS_DIR], file_path)
    return file_path


X = TypeVar("X")


def pairwise_iterate(iterable: Iterable[X]) -> Iterable[Tuple[X, X]]:
    """
    s -> (s0, s1), (s2, s3), (s4, s5), ...
    :param iterable: s (list of X)
    :return: (s0, s1), (s2, s3), (s4, s5) (iterate over all X in groups of 2, without sliding window).
    """
    # https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    a = iter(iterable)
    return zip(a, a)


def group_by_dial_id_and_turn(turns: Iterable[Dict], dialogue_id_key: str = "ID", turn_id_key: str = "turn_id") -> Dict[str, List[Dict]]:
    result = defaultdict(dict)
    for turn in turns:
        result[turn[dialogue_id_key]][turn[turn_id_key]] = turn
    return {dial_id: [turn for index, turn in sorted(turns_dict.items(), key=lambda item: item[0])]
            for dial_id, turns_dict in result.items()}


def print_dialogue(dialogue: RawMultiWOZDialogue, up_to_turn: int = None) -> None:
    for i, log in enumerate(dialogue['log']):
        if up_to_turn and i // 2 > up_to_turn:
            break
        prefix = "system: " if log['metadata'] else "user: "
        print(prefix + log['text'])


def is_jsonable(x: Any) -> bool:
    """
    Return whether the given object is json serializable acording the current json module context
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    root_logger.setLevel(logging.INFO)


def clean_string_for_fuzzy_comparison(match_or_query: str) -> str:
    # lowercase, remove punctuation, and remove extra whitespace
    return match_or_query.lower().translate(str.maketrans('', '', string.punctuation)).strip()
