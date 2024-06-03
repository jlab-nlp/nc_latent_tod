import abc
from typing import List

from nc_latent_tod.db.types import DBEntity
from nc_latent_tod.data_types import ServiceBeliefState


class AbstractDB(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def query(self, service_name: str, constraints: ServiceBeliefState, fuzzy_ratio: int = 90) -> List[DBEntity]:
        """
        Queries the database based on the given service name and constraints (informable slots of that service which
        can be used to filter the known entities in that service)

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
        pass

