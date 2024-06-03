import unittest
from typing import List, Set, Any, Dict

from datasets import load_dataset
from tqdm import tqdm

from nc_latent_tod.db.multiwoz_db import MultiWOZDB
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.utils.dialogue_states import get_mwzeval_db_slot_name
from nc_latent_tod.utils.dialogue_states import remove_blank_values
from nc_latent_tod.utils.general import DELETE_VALUE


class MultiWOZDBTests(unittest.TestCase):

    db: MultiWOZDB
    schema: List[ServiceSchema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.db = MultiWOZDB()
        cls.schema = read_multiwoz_schema()

    def test_dataset_slot_names_are_filterable(self):
        dataset = load_dataset("Brendan/icdst_multiwoz_turns_v24")
        evaluable_service_names: Set[str] = set(s['service_name'] for s in self.schema)
        for split in dataset:
            for turn in tqdm(dataset[split], desc=f"validating turns in {split}"):
                turn_slot_values = remove_blank_values(turn['turn_slot_values'])
                deletes_only: List[str] = []
                for service in turn_slot_values:
                    if all(value == DELETE_VALUE for value in turn_slot_values[service].values()):
                        deletes_only.append(service)
                turn_slot_values = {k: v for k, v in turn_slot_values.items() if k not in deletes_only}
                slot_values = remove_blank_values(turn['slot_values'])

                # inferring these to be active domains
                for service in turn_slot_values:
                    if service in evaluable_service_names:
                        constraints = {get_mwzeval_db_slot_name(slot_name=slot): value
                                       for slot, value in slot_values[service].items()}
                        for slot, value in constraints.items():
                            if not slot.startswith('book') and not service == 'taxi':
                                self.assertIn(slot, self.db.db.data_keys[service])

    def test_basic_search(self):
        constraints = {"area": "centre", "type": "hotel"}
        results: List[Dict[str, Any]] = self.db.query('hotel', constraints)
        self.assertGreaterEqual(len(results), 1)
        expected_keys: Set[str] = set(results[0].keys())
        for hotel in results:
            self.assertEqual(hotel['area'], "centre")
            self.assertEqual(hotel['type'], "hotel")
            self.assertSetEqual(expected_keys, set(hotel.keys()))


if __name__ == '__main__':
    unittest.main()
