from typing import TypedDict, Optional, List, Dict


class SlotSchema(TypedDict):
    name: str  # slot name (e.g. hotel-area)
    description: str  # description of that slot e.g. 'area or place of the hotel'
    possible_values: Optional[List[str]]  # accepted values (for is_categorical=True) e.g. ['centre', 'east', ...]
    is_categorical: bool  # whether the slot values are categorical/pre-defined


class IntentSchema(TypedDict):
    name: str  # e.g. find_hotel or book_hotel
    description: str  # e.g. 'search for a hotel to stay in' or 'book a hotel to stay in'
    required_slots: Optional[List[str]]  # slots which must have a value specified
    optional_slots: Optional[Dict[str, str]]  # slots which may, and their defaults when they are not specified
    is_transactional: bool  # is use of the API is stateful


class ServiceSchema(TypedDict):
    service_name: str  # service/domain name e.g. hotel
    slots: List[SlotSchema]  # list of defined slots (name, desc, possible values, etc.)
    description: str  # description of the service, e.g. 'hotel reservations and vacation stays'
    intents: List[IntentSchema]  # actions available in that domain (in MultiWOZ typically find_X and book_X)
