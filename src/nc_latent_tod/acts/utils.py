import importlib
import json
import logging
from collections import defaultdict
from types import ModuleType
from typing import Any, Type, Callable, Set

# Important! Do not import less than *, since in get_acts_from_system_acts we use globals() to find the act classes
from nc_latent_tod.acts.act_definitions import *
from nc_latent_tod.data_types import ValidActsRepresentation, SystemActBatch
from nc_latent_tod.kwargs_prompt.templates import SERVICE_ENTITY_TEMPLATE
from nc_latent_tod.kwargs_prompt.utils import build_docstring
from nc_latent_tod.prompting.abstract_prompt_generator import clean_slot_name, get_type_hint
from nc_latent_tod.schemas.data_types import ServiceSchema

_ACT_TYPE_TO_CONSTRUCTOR: Dict[str, Callable[[Type, Dict[str, Any]], ServiceAct]] = {
    # We receive an entity type and slot_pairs arguments: construct entity in the act class
    "inform": lambda entity_type, kwargs: Inform(entity=entity_type(**kwargs)),
    "request": lambda entity_type, kwargs: Request(service=entity_type.__name__.lower(),
                                                   values=[k for k in kwargs if k != 'service']),
    "recommend": lambda entity_type, kwargs: Offer(entity=entity_type(**kwargs)),
    "offerbook": lambda entity_type, kwargs: Confirm(entity=entity_type(**kwargs)),
    "offerbooked": lambda entity_type, kwargs: NotifySuccess(service=entity_type.__name__.lower(),
                                                             entity=entity_type(**kwargs) if kwargs else None),
    "nooffer": lambda entity_type, kwargs: NotifyFailure(service=entity_type.__name__.lower(),
                                                         entity=entity_type(**kwargs) if kwargs else None),
    "reqmore": lambda entity_type, kwargs: RequestAlternatives(),
    "book": lambda entity_type, kwargs: NotifySuccess(service=entity_type.__name__.lower(),
                                                      entity=entity_type(**kwargs) if kwargs else None),
    "bye": lambda entity_type, kwargs: Goodbye(),
    "welcome": lambda entity_type, kwargs: Greeting(),
    "select": lambda entity_type, kwargs: Offer(entity=entity_type(**kwargs)),
    "general": lambda entity_type, kwargs: Greeting(),
    "greet": lambda entity_type, kwargs: Greeting(),
    "nobook": lambda entity_type, kwargs: NotifyFailure(service=entity_type.__name__.lower(),
                                                        entity=entity_type(**kwargs) if kwargs else None),
    "thank": lambda entity_type, kwargs: ThankYou(),
}


def get_acts_from_system_acts(acts: ValidActsRepresentation, schema: List[ServiceSchema],
                              unfill_act_values: bool = False,
                              act_loading_context: Dict[str, Any] = None) -> List[Act]:
    """
    Convert a list of system acts into a list of ServiceAct classes.

    :param acts: list of system acts (slots are not grouped by act or service)
    :param unfill_act_values: if True, will replace value in a service act with a placeholder, e.g. [value_price]
    :return: list of ServiceAct classes (slots are grouped by act and service)
    """
    if not acts:
        return []
    loading_context: Dict[str, Any] = act_loading_context or get_act_loading_context(schema)
    possible_batch: SystemActBatch = defaultdict(list)
    if len(acts) == 0:
        return []  # could have been dict for SystemActBatch, but we will just return empty list for an empty batch
    if isinstance(acts, dict):
        acts: SystemActBatch = acts
        # group by act and service:
        act_groups = defaultdict(dict)
        for act, service, slot_name, value in zip(acts['act'], acts['service'], acts['slot_name'], acts['value']):
            if any([act, service, slot_name, value]):
                act_groups[(act, service)][slot_name.strip()] = value.strip()
        # convert to classes:
        act_classes: List[ServiceAct] = []
        for (act, service), slot_pairs in act_groups.items():
            # filter out slot names that are empty or 'none'
            slot_pairs = {k: v for k, v in slot_pairs.items() if v and k.lower() != 'none'}
            # if any values are 'none', check if their slot is a boolean and change to 'yes' if so
            for slot_name, value in slot_pairs.items():
                if value.lower() == 'none':
                    slot_pairs[slot_name] = 'yes'
            # convert slot_names if they don't match the schema
            slot_pairs = {convert_to_schema_slot_name(k, service_name=service): v for k, v in slot_pairs.items()}
            service_entity_type: Type = loading_context.get(service.capitalize(), None)
            if service == 'booking':
                # MultiWOZ contains a 'booking' service that is generically used across bookings. We don't use this
                # since our system doesn't see act annotations, but we'll fall back to a generic Entity for
                # development slot/value 'f1' analysis.
                service_entity_type = Entity
            service_act: ServiceAct = _ACT_TYPE_TO_CONSTRUCTOR[act.lower()](service_entity_type, slot_pairs)
            act_classes.append(service_act)
        return act_classes
    elif isinstance(acts, list):
        result_acts: List[Act] = []
        for act in acts:
            if isinstance(act, Act):
                result_acts.append(act)
            elif isinstance(act, str):
                if act == "":
                    # sometimes inserted to not upset pyarrow when used as an hf dataset
                    continue
                try:
                    dict_act: Dict[str, Any] = json.loads(act)
                except json.JSONDecodeError as e:
                    if not e.msg == 'Expecting property name enclosed in double quotes':
                        logging.warning(f"Could not parse act {act} as json: {e}")
                    # For whatever reason, the huggingface datasets sends us back a single-quoted dict. It could have
                    # been written this way in the original dataset. We will try to parse it as a dict by evaluating:
                    dict_act: Dict[str, Any] = eval(act)
                    assert dict_act or not act, "if the string is not empty, it should be parsable as a dict"
                # now one more decision: if it's a string WE serialized, it will have __type as a key, and we can call
                # its class by name. Otherwise, we'll add it to a 'batch' of system acts, and append these via a
                # recursive call hitting the dict case above.
                if "__type" in dict_act:
                    act_type: str = dict_act.pop("__type")
                    if not act_type or act_type not in globals():
                        raise ValueError(f"Unknown act type {act_type}, make sure it is imported in this class if "
                                         f"needed, or that the data is properly formatted: {act}")
                    act_class: Type[Act] = globals()[act_type]
                    # we pass in our globals in case there is any nesting, e.g. an Inform act has
                    # a Hotel entity as an argument
                    result_acts.append(act_class.from_dict(dict_act, loading_context=loading_context))
                else:
                    # could be in original MultiWOZ format
                    for key in ('act', 'service', 'slot_name', 'value'):
                        if key not in dict_act:
                            raise ValueError(f"Could not parse act, missing key: {key}, "
                                             f"possibly unknown format: {act}")
                        possible_batch[key].append(dict_act[key])
        if len(possible_batch) > 0:
            # call should hit dict case above
            result_acts.extend(get_acts_from_system_acts(possible_batch, schema, act_loading_context=loading_context))

        if unfill_act_values:
            for act in result_acts:
                if type(act) == Request:
                    continue
                for slot_name, value in vars(act).items():
                    if isinstance(value, Entity):
                        for entity_slot_name, entity_slot_value in vars(value).items():
                            if entity_slot_name not in ('service', 'act') and entity_slot_value is not None:
                                setattr(value, entity_slot_name, f"[value_{entity_slot_name}]")
                    elif slot_name not in ('service', 'act') and value is not None:
                        setattr(act, slot_name, f"[value_{slot_name}]")
        return result_acts


def get_act_loading_context(schema: List[ServiceSchema]) -> Dict[str, Any]:
    service_entity_classes: List[str] = get_act_entity_classes_from_schema(schema)
    loading_context: Dict[str, Any] = {"Entity": Entity, "Act": Act}
    # add the remaining act definitions by importing, excluding private/python builtins
    act_definitions: ModuleType = importlib.import_module("nc_latent_tod.acts.act_definitions")
    loading_context.update({k: v for k, v in act_definitions.__dict__.items() if not k.startswith("__")})
    for service_entity in service_entity_classes:
        exec(service_entity, loading_context)
    return loading_context


def get_act_entity_classes_from_schema(schema: List[ServiceSchema]) -> List[str]:
    """
    For each service in the schema, produce a nc_latent_tod.acts.act.Entity sub-class for discussing entities in that service
    (e.g. Hotel, Restaurant, etc.). Different from DST, these will also user requestable slots (every slot in schema).
    These can be included in a prompt but are also executable for parsing.
    """
    service_entity_classes: List[str] = []
    for service_schema in schema:
        service_name: str = service_schema['service_name']

        # get a docstring for the class
        docstring: str = build_docstring(service_schema, informable_only=False, intent=None)

        # e.g: "price_range: str = None  # 'expensive', 'cheap', 'moderate', 'dontcare'"
        argument_strings = [
            # start with <clean_slot_name>: <type_hint> = None
            f"{clean_slot_name(slot['name'], service_name)}: {get_type_hint(slot)}"
            for slot in service_schema['slots']
        ]
        service_entity: str = SERVICE_ENTITY_TEMPLATE.format(
            service_class_name=service_name.capitalize(),
            docstring=docstring,
            parameters="\n    ".join(argument_strings))
        service_entity_classes.append(service_entity)
    return service_entity_classes


__ALL_MULTIWOZ_SCHEMA_SLOT_NAMES: Set[str] = {
    'arrive by', 'name', 'postcode', 'book stay', 'price', 'ref', 'leave at', 'stars', 'address', 'departure',
    'price range', 'area', 'entrancefee', 'destination', 'type', 'book day', 'trainid', 'phone', 'book time',
    'book people', 'openhours', 'day', 'food', 'internet', 'duration', 'parking',
    # Brendan additions
    'choice', 'department'
}


def convert_to_schema_slot_name(slot_name: str, service_name: str) -> str:
    SERVICE_SPECIFIC_REPLACEMENTS: Dict[str, Dict[str, str]] = {
        "train": {"time": "duration", "id": "trainid"},
        "attraction": {"fee": "entrancefee", "open": "openhours"},
        "taxi": {"car": "type"},
        "booking": {"time": "book time"}
    }
    REPLACEMENTS: Dict[str, str] = {
        "addr": "address", "leave": "leave at", "arrive": "arrive by", "ticket": "price", "depart": "departure",
        "dest": "destination", "people": "book people", "post": "postcode", "stay": "book stay",
        "pricerange": "price range",
        "leaveat": "leave at", "arriveby": "arrive by", "bookstay": "book stay", "bookpeople": "book people",
        "booktime": "book time", "bookday": "book day"

    }
    cleaned_slot_name: str = SERVICE_SPECIFIC_REPLACEMENTS.get(service_name, {}).get(slot_name, slot_name)
    cleaned_slot_name: str = REPLACEMENTS.get(cleaned_slot_name, cleaned_slot_name)
    assert cleaned_slot_name in __ALL_MULTIWOZ_SCHEMA_SLOT_NAMES, f"{service_name}, {cleaned_slot_name}"
    return cleaned_slot_name


def _read_json_act_for_key(act: str) -> Dict[str, Any]:
    try:
        return json.loads(act)
    except json.JSONDecodeError as e:
        if not e.msg == 'Expecting property name enclosed in double quotes':
            logging.warning(f"Could not parse act {act} as json: {e}")
        # For whatever reason, the huggingface datasets sends us back a single-quoted dict. It could have
        # been written this way in the original dataset. We will try to parse it as a dict by evaluating:
        try:
            dict_act: Dict[str, Any] = eval(act)
            return dict_act
        except Exception as e:
            logging.error(f"Could not parse act {act} as json or dict: {e}")
            return {}


def get_acts_slots_as_key_string(acts: List[str]) -> str:
    """
    Returns a string which uniquely identifies a list of acts by the non-empty act & slot names
    """
    all_values: List[str] = []
    for act in acts:
        act_dict: Dict[str, Any] = _read_json_act_for_key(act)
        type_str: str = act_dict.get("__type", "unknown")
        if 'entity' in act_dict:
            entity_type_str: str = act_dict['entity']['__type']
            for k, v in act_dict['entity'].items():
                if k not in ('entity', '__type') and v:
                    all_values.append(f"{type_str}.{entity_type_str}.{k}")
        # add outer values
        all_values.extend([f"{type_str}.{k}" for k in act_dict.keys() if k not in ('entity', '__type') and act_dict[k]])
    return "-".join(sorted(all_values))
