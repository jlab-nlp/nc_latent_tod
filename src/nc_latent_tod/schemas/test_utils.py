import unittest
from typing import Dict, List

from nc_latent_tod.schemas.data_types import IntentSchema

from nc_latent_tod.schemas.utils import get_all_intents
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.utils.testing import test_suite


@test_suite("unit_build")
class MyTestCase(unittest.TestCase):
    def test_expected_multiwoz_intents(self):
        schema = read_multiwoz_schema()
        all_intents: Dict[str, List[IntentSchema]] = get_all_intents(schema)
        for service, intents in all_intents.items():
            self.assertLessEqual(len(intents), 1, msg=f"too many intents for {service}")
            if intents:
                if service == 'taxi':
                    self.assertEqual(intents[0]['name'], 'book_taxi')
                else:
                    self.assertEqual(intents[0]['name'], f'find_{service}')
            else:
                print(f"no intents for {service}")


if __name__ == '__main__':
    unittest.main()
