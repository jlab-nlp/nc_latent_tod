import unittest
from typing import List

from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.schemas.data_types import ServiceSchema


class SchemaReaderTests(unittest.TestCase):
    def test_multiwoz(self):
        multiwoz: List[ServiceSchema] = read_multiwoz_schema(only_evaluated_schemas=False)
        service_names: List[str] = sorted([service['service_name'] for service in multiwoz])
        self.assertListEqual(
            service_names, ["attraction", "bus", "hospital", "hotel", "police", "restaurant", "taxi", "train"]
        )

    def test_multiwoz_has_all_intent_slots(self):
        multiwoz: List[ServiceSchema] = read_multiwoz_schema()
        all_slot_names = set()
        for service in multiwoz:
            all_slot_names.update(slot['name'] for slot in service['slots'])
        for service in multiwoz:
            for intent in service['intents']:
                for slot_name in intent['required_slots']:
                    self.assertIn(slot_name, all_slot_names)
                for slot_name in intent['optional_slots']:
                    self.assertIn(slot_name, all_slot_names)


if __name__ == '__main__':
    unittest.main()
