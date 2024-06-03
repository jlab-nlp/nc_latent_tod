from typing import List, Callable, Optional, Any, Set

from datasets import Dataset, Features
from transformers import AutoTokenizer

from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.retrieval.mpnet.mpnet_model_for_retrieval import MPNetModelForRetrieval
from nc_latent_tod.retrieval.mpnet.mpnet_retriever import MPNetRetriever, get_context_system_user_encoding


class DistinctMPNetRetriever(MPNetRetriever):

    overfetch: int  # how many items to fetch regardless of k, to then filter k distinct elements from
    minimum_distinct: Optional[int]
    distinguishing_fn: Callable[[DatasetTurn], Any]

    def __init__(self, model: MPNetModelForRetrieval = None, tokenizer: AutoTokenizer = None, dataset: Dataset = None,
                 features: Features = None, model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2",
                 train_index: bool = False, init_on_cuda: bool = False,
                 turn_encoding_context_fn: Callable[[DatasetTurn], str] = get_context_system_user_encoding,
                 overfetch: int = 10,
                 minimum_distinct: Optional[int] = None,
                 distinguishing_fn: Callable[[DatasetTurn], Any] = repr,
                 **kwargs) -> None:
        super().__init__(model, tokenizer, dataset, features, model_name_or_path, train_index, init_on_cuda,
                         turn_encoding_context_fn, **kwargs)
        self.overfetch = overfetch
        self.minimum_distinct = minimum_distinct
        self.distinguishing_fn = distinguishing_fn

    def get_nearest_examples_batched(self, turns: List[DatasetTurn], k: int = 10) -> List[List[DatasetTurn]]:
        total_examples: List[List[DatasetTurn]] = super().get_nearest_examples_batched(turns, max(self.overfetch, k))
        minimum_distinct: int = self.minimum_distinct or k

        def first_k_distinct(turns: List[DatasetTurn]) -> List[DatasetTurn]:
            distinct: List[DatasetTurn] = []
            seen: Set[Any] = set()
            seen_indices: Set[int] = set()
            for i, turn in enumerate(turns):
                distinguishing_value: Any = self.distinguishing_fn(turn)
                if distinguishing_value not in seen:
                    distinct.append(turn)
                    seen.add(distinguishing_value)
                    seen_indices.add(i)
                if len(distinct) == minimum_distinct:
                    break
            # len(distinct) still is less than k: add the first k - len(distinct) turns that we haven't 'seen' yet
            i: int = 0
            while len(distinct) < k:
                if i not in seen_indices:
                    distinct.append(turns[i])
                i += 1
            return distinct
        return [first_k_distinct(turns) for turns in total_examples]

    def load_faiss_index(self):
        return super().load_faiss_index()
