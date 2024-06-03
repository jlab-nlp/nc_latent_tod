import re
from typing import List

from nc_latent_tod.resources import read_json_resource
from nc_latent_tod.schemas.data_types import ServiceSchema


def clean_slot_name(slot_name: str, service_name: str) -> str:
    slot_name = slot_name.lower().replace('semi-', '').replace('book-', 'book ').replace('pricerange', 'price range')
    slot_name = slot_name.replace('leaveat', 'leave at').replace('arriveby', 'arrive by')
    # insert a space after book for any bookx where x is non-whitespace
    slot_name = re.sub(r'book(\S)', lambda match: f'book {match.group(1)}', slot_name)
    return re.sub(service_name + r'[-_]', '', slot_name)


def clean_multiwoz_service_schema(service_schema: ServiceSchema) -> ServiceSchema:
    service_name: str = service_schema['service_name']
    for slot in service_schema['slots']:
        # force agreement of slot names between schema and convlab
        slot['name'] = clean_slot_name(slot['name'], service_name)
        # this should actually be a yes/no boolean
        if set(slot.get('possible_values', [])) == {'yes', 'no', 'free'}:
            slot['possible_values'] = ['yes', 'no']
        if 'possible_values' in slot and 'dontcare' not in slot['possible_values']:
            slot['possible_values'].append('dontcare')
    for intent in service_schema['intents']:
        intent['required_slots'] = [clean_slot_name(slot_name, service_name) for slot_name in intent['required_slots']]
        intent['optional_slots'] = {
            clean_slot_name(slot_name, service_name): default_value
            for slot_name, default_value in intent['optional_slots'].items()
        }
    return service_schema


def read_multiwoz_schema(only_evaluated_schemas: bool = True) -> List[ServiceSchema]:
    """
    Returns the service schema's in the MultiWOZ benchmark

    :return: list of service schemas
    """
    schema: List[ServiceSchema] = read_json_resource("schemas/data/multiwoz.json")
    return [
        clean_multiwoz_service_schema(service) for service in schema
        if not only_evaluated_schemas or service['service_name'] in (
            'attraction', 'hotel', 'restaurant', 'taxi', 'train'
        )
    ]


def read_tm4_schema() -> List[ServiceSchema]:
    schema: List[ServiceSchema] = read_json_resource("schemas/data/taskmaster4.json")
    return schema


if __name__ == '__main__':
    print(read_tm4_schema())