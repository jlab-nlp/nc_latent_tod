from collections import defaultdict
from typing import List, Set, Dict

from nc_latent_tod.schemas.data_types import ServiceSchema, SlotSchema, IntentSchema


def get_informable_slots(service_schema: ServiceSchema, intents: List[IntentSchema] = None) -> List[SlotSchema]:
    """
    Return the subset of slots (SlotSchemas) which are informable: these are any slot that can be used as an optional
    or required argument in an intent (IntentSchemas). I.e. informable by user in an 'inform' dialogue act.

    :param service_schema: a service schema composed of slots and intents
    :param intents: Optional, the intents to consider when determining informable slots. If None, use all intents in the
            service schema.
    :return: the subset of all slots that can be used as arguments to an intent
    """
    informable_slot_names: Set[str] = set()
    for intent in (intents or service_schema['intents']):
        informable_slot_names.update(intent['required_slots'])
        informable_slot_names.update(intent['optional_slots'].keys())
    return [slot for slot in service_schema['slots'] if slot['name'] in informable_slot_names]


def get_all_informable_slot_names(schema: List[ServiceSchema]) -> Dict[str, List[str]]:
    all_informable_slots: Dict[str, List[str]] = {}
    for service_schema in schema:
        all_informable_slots[service_schema['service_name']] = [
            slot['name'] for slot in get_informable_slots(service_schema)
        ]
    return all_informable_slots


def get_intent_argument_signature(intent: IntentSchema) -> str:
    """
    Returns a unique key listing the arguments of an intent. This is used to determine if two intents are the same
    in all but name and is_transactional
    """
    return f"req={','.join(intent['required_slots'] or [])}, opt={','.join(intent['optional_slots'].keys() or [])}"


def get_all_intents(schema: List[ServiceSchema], allow_duplicate_signatures: bool = False) -> Dict[str, List[IntentSchema]]:
    """
    Return a mapping from service name to a list of IntentSchema for that service. If allow_duplicate_signatures is
    False, then only return the first intent with a given signature (required_slots, optional_slots).
    """
    all_intents: Dict[str, List[IntentSchema]] = defaultdict(list)

    seen_signatures: Dict[str, Set[str]] = defaultdict(set)
    for service_schema in schema:
        service_name: str = service_schema['service_name']
        for intent in service_schema['intents']:
            signature: str = get_intent_argument_signature(intent)
            if allow_duplicate_signatures or signature not in seen_signatures[service_name]:
                all_intents[service_name].append(intent)
                seen_signatures[service_name].add(signature)
    return all_intents

