import unittest
from typing import List

from nc_latent_tod.acts.act import Entity, Act
from nc_latent_tod.acts.act_definitions import Confirm
from nc_latent_tod.normalization.schema_normalizer import SchemaNormalizer
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.utils.testing import test_suite


class ExperimentLogManifest:
    pass


@test_suite("unit_build")
class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        schema = read_multiwoz_schema()
        cls.schema = schema
        cls.schema_normalizer = SchemaNormalizer(schema=schema)

    def test_simple_normalize(self):
        self.assertEqual(
            {"hotel": {"name": "grand budapest hotel"}},
            self.schema_normalizer.normalize({"hotel": {"name": "grand budapest hotel", "area": "norwich"}}),
        )

    def test_simple_normalize_acts(self):
        acts: List[Act] = [
            Confirm(response='some long text', entity=Entity(service_name='hotel', name='grand budapest hotel'))
        ]
        normalized_acts = self.schema_normalizer.normalize_acts(acts)
        self.assertListEqual(
            normalized_acts,
            [Confirm(entity=Entity(name='grand budapest hotel'))]
        )


if __name__ == '__main__':
    unittest.main()
