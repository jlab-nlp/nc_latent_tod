import abc
import re
from abc import ABC
from typing import List

from num2words import num2words

from nc_latent_tod.schemas.data_types import ServiceSchema


class AbstractOntology(abc.ABC):
    """
    An abstract class for general expectations we might have of an ontology, such
    as determining whether a slot name is valid, whether a slot is categorical, etc. In
    this broadest case, we won't assume access to the underlying DB of entities, which could
    give information about non-categorical slot values.
    """

    @abc.abstractmethod
    def __init__(self, schema: List[ServiceSchema], min_fuzzy_match: int = 95, **kwargs):
        pass

    # separating for testing
    @staticmethod
    def is_valid_time(value: str) -> bool:
        return bool(re.match(r"^([0-1]?\d|2[0-4]):[0-5]\d$", value))

    @staticmethod
    def _per_digit_num2words(token: str) -> str:
        if len(token) > 1 and token.isnumeric():
            return ' '.join(num2words(digit) for digit in token)
        else:
            return num2words(token)

    @abc.abstractmethod
    def is_categorical(self, service_name: str, slot_name: str) -> bool:
        pass

    @abc.abstractmethod
    def is_numeric(self, service_name: str, slot_name: str) -> bool:
        pass

    @abc.abstractmethod
    def is_bool(self, service_name: str, slot_name: str) -> bool:
        pass


class AbstractDBOntology(AbstractOntology, ABC):
    """
    An ontology in which DB information (i.e. values for non-categorical slots) is also available. This does not need
    to include dialogue data.
    """

    @abc.abstractmethod
    def get_canonical(self, service_name: str, slot_name: str, slot_value: str) -> str:
        """
        Given a slot name and a slot value, return the canonical form of that slot value. For categorical slots, it
        should be one of the possible_values in the schema. For non-categorical slots, it should be the value associated
        with the entity in the database (e.g. the true hotel name)

        :param service_name: name of the service in the schema (e.g. 'hotel')
        :param slot_name: name of the slot within the service (e.g. 'name')
        :param slot_value: surface form to query associated with slot_name (e.g. 'acron guest house')
        :return: canonical slot value (e.g. 'the acorn guest house')
        """
        pass

    def is_name(self, service_name: str, slot_name: str) -> bool:
        pass

    def is_valid_slot(self, service_name: str, slot_name: str) -> bool:
        """
        Return true if the slot name exists in the schema associated with this ontology, false otherwise.

        :param service_name: name of the service (e.g. 'hotel')
        :param slot_name: slot name within service (e.g. 'name')
        :return: True if in ontology, False otherwise.
        """
        pass
