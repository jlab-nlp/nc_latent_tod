import abc
from typing import List, Union

from datasets import Dataset

from nc_latent_tod.data_types import DatasetTurn


class AbstractRetriever(metaclass=abc.ABCMeta):
    """
    Abstract class for in-context example retrievers
    """
    @abc.abstractmethod
    def get_nearest_examples_batched(self, turns: List[DatasetTurn], k: int = 10) -> List[List[DatasetTurn]]:
        """
        Return the k nearest turns in the dataset according to the retriever. If k is greater
        than the number of turns in the dataset, returning the complete dataset.

        :param turns: turns to use as query
        :param k: number of nearest turns to return per turn in turns
        :return: grouped list of nearest turns
        """
        pass

    @abc.abstractmethod
    def add_items(self, turns: Union[List[DatasetTurn], Dataset]) -> None:
        """
        Add turns to the retriever's index

        :param turns: turns to add
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()
