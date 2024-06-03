import random
from typing import Union, List

from datasets import Dataset

from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.retrieval.abstract_retriever import AbstractRetriever


class RandomRetriever(AbstractRetriever):

    pool: List[DatasetTurn]

    def __init__(self, pool: Union[List[DatasetTurn], Dataset] = None) -> None:
        super().__init__()
        if type(pool) == Dataset:
            self.pool = pool.to_list()
        else:
            self.pool = pool or []

    def get_nearest_examples_batched(self, turns: List[DatasetTurn], k: int = 10) -> List[List[DatasetTurn]]:
        """
        Return k random examples sampled from the index without replacement, for each turn

        :param turns: turns to use as query
        :param k: number of random turns to retrieve per turn in turns
        :return: 2D array of turns (len(turns) x k)
        """

        return [random.sample(self.pool, k=min(len(self.pool), k)) for _ in turns]

    def add_items(self, turns: Union[List[DatasetTurn], Dataset]) -> None:
        """
        Add turns to the retriever's index

        :param turns: turns to add
        """
        self.pool.extend(turns)

    def __len__(self):
        return len(self.pool)
