import abc
from typing import List

from nc_latent_tod.acts.act import Act


class AbstractDelexer(metaclass=abc.ABCMeta):
    """
    This module takes in the system acts with slots filled (predicted by the system dialog act tracker, 2) and the
    system response observed in the data and produces a de-lexicalized sys- tem response. This de-lexicalized
    response is used to train the response generator in the trained system, and an evaluable response in the
    un-trained system. This could be rule- based or a prompted LLM.
    """

    @abc.abstractmethod
    def delexify(self, lexicalized_response: str, system_acts: List[Act], transform_slots_for_eval: bool = False, **kwargs) -> str:
        pass
