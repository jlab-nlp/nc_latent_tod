import json
import re
import sys
from typing import List, Dict


def clean_slot_name(slot_name: str, service_name: str) -> str:
    slot_name = slot_name.lower().replace('semi-', '').replace('book-', 'book ').replace('pricerange', 'price range')
    slot_name = slot_name.replace('leaveat', 'leave at').replace('arriveby', 'arrive by')
    # insert a space after book for any bookx where x is non-whitespace
    slot_name = re.sub(r'book(\S)', lambda match: f'book {match.group(1)}', slot_name)
    return re.sub(service_name + r'[-_]', '', slot_name)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data: Dict[str, List[str]] = json.load(f)

    new_ontology: Dict[str, Dict[str, List[str]]] = {}
    for full_slot_name, values in data.items():
        service_name, slot_name = full_slot_name.split("-")
        slot_name = clean_slot_name(slot_name=slot_name, service_name=service_name)
        if service_name not in new_ontology:
            new_ontology[service_name] = {}
        new_ontology[service_name][slot_name] = values

    with open(sys.argv[1], 'w') as f:
        json.dump(new_ontology, f, indent=4)
