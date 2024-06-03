import pprint
from collections import defaultdict
from typing import Dict, Any, List

from nc_latent_tod.mwzeval.database import MultiWOZVenueDatabase

from nc_latent_tod.db.abstract_db import AbstractDB
from nc_latent_tod.utils.dialogue_states import get_mwzeval_db_slot_name


class MultiWOZDB(AbstractDB):
    """
    A class to represent the MultiWOZ database.

    Attributes:
    -----------
    db : MultiWOZVenueDatabase
        An instance of the MultiWOZVenueDatabase.
    id_to_entity : Dict[str, Dict[str, Any]]
        A mapping from service names to entities indexed by their IDs.
    """
    db: MultiWOZVenueDatabase
    id_to_entity: Dict[str, Dict[str, Any]]

    def __init__(self) -> None:
        super().__init__()
        self.db = MultiWOZVenueDatabase()
        self.id_to_entity = defaultdict(dict)

        for service in self.db.data:
            for entity in self.db.data[service]:
                id_key: str = 'trainid' if service == 'train' else 'id'
                assert id_key in entity, f"expected {pprint.pformat(entity)} to have an id under {id_key}"
                self.id_to_entity[service][entity[id_key]] = entity

    def query(self, service_name: str, constraints: Dict[str, Any], fuzzy_ratio: int = 90) -> List[Dict[str, Any]]:
        """
        Queries the database based on the given service name and constraints (informable slots of that service which
        can be used to filter the known entities in that service). Wraps the DB implementation in
        Tomiinek/MultiWOZ_Evaluation, ensuring accurate inform/success

        Parameters:
        -----------
        service_name : str
            The service/domain to query.
        constraints : Dict[str, Any]
            A dictionary of constraints to filter the entities in that service with.
        fuzzy_ratio : int, optional
            The fuzzy matching ratio for the query, used for values like `name`. Default is 90.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of entities that match the given constraints.
        """
        # clean slot names to what's expected by the DB
        constraints = {
            get_mwzeval_db_slot_name(slot): value
            # ignore empty values
            for slot, value in constraints.items() if value
        }
        entity_ids: List[str] = self.db.query(domain=service_name, constraints=constraints, fuzzy_ratio=fuzzy_ratio)
        return [self.id_to_entity[service_name][e_id] for e_id in entity_ids]


if __name__ == '__main__':
    db = MultiWOZDB()
    for service in db.db.data:
        print(f"service={service}")
        pprint.pprint(db.db.data[service][0])
