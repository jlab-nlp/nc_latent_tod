import logging
from multiprocessing import cpu_count
from typing import Union, List, TypedDict, Tuple, Optional, Callable

import faiss
import numpy.typing as npt
import torch
from datasets import Dataset, Features
from datasets.search import FaissIndex, SearchResults
from nc_latent_tod.utils.arrow_list import ArrowList
from torch import Tensor
from transformers import AutoTokenizer

from nc_latent_tod.data_types import DatasetTurn, ServiceBeliefState
from nc_latent_tod.retrieval.abstract_retriever import AbstractRetriever
from nc_latent_tod.retrieval.mpnet.mpnet_model_for_retrieval import MPNetModelForRetrieval
from nc_latent_tod.utils.dialogue_states import remove_blank_values


class HasEncoderInput(TypedDict):
    encoder_input: List[str]  # un-tokenized input to mpnet/encoder


def get_context_system_user_encoding(turn: DatasetTurn) -> str:
    # same content as our prior work, but differently formatted
    context_service_strings: List[str] = [
        _get_service_string(service, state)
        for service, state in remove_blank_values(turn['last_slot_values']).items()
    ]
    context_str: str = "Context: " + " ".join(context_service_strings)
    system_utt = turn['system_utterances'][-1]
    system_str = "System: " + system_utt if system_utt != 'none' else ''
    user_str = "User: " + turn['user_utterances'][-1]
    result: str = "\n".join([context_str, system_str, user_str])
    return result


def _get_service_string(service_name: str, service_state: ServiceBeliefState) -> str:
    slot_strings: List[str] = [f"{slot} is {value}" for slot, value in service_state.items() if value]
    return f"for {service_name}, the {', '.join(slot_strings)} ."


def map_to_encoder_input(turn, turn_encoding_context_fn: Callable[[DatasetTurn], str] = get_context_system_user_encoding) -> HasEncoderInput:
    return {"encoder_input": turn_encoding_context_fn(turn)}


class MPNetRetriever(AbstractRetriever):
    """
    Composes the model and tokenizer in order to facilitate common retrieval use-cases
    """

    model_name_or_path: str
    model: MPNetModelForRetrieval
    tokenizer: AutoTokenizer
    dataset: Optional[ArrowList]
    features: Features
    # I found it more flexible to manage my own vs dataset.add_faiss_index. Looking at the method, it essentially
    # constructs one and has niceties for converting back to dataset items
    faiss_index: FaissIndex
    train_index: bool
    turn_encoding_context_fn: Callable[[DatasetTurn], str]

    def __init__(self,
                 model: MPNetModelForRetrieval = None,
                 tokenizer: AutoTokenizer = None,
                 dataset: Dataset = None, features: Features = None,
                 model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2",
                 train_index: bool = False,
                 init_on_cuda: bool = False,
                 turn_encoding_context_fn: Callable[[DatasetTurn], str] = get_context_system_user_encoding,
                 **kwargs) -> None:
        """
        Can be instantiated with just a model_name_or_path (to load from HF) or give a model that's already been
        instantiated. Similarly, a dataset can be given, or just features, where a dataset and index are built after
        items are added.
        """
        super().__init__()
        if not features and not dataset:
            raise ValueError("must set either the dataset, or the features it should have when items are added")
        self.train_index = train_index
        self.turn_encoding_context_fn = turn_encoding_context_fn
        if not model:
            self.model_name_or_path = model_name_or_path
            self.model = MPNetModelForRetrieval.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model_name_or_path = model.name_or_path
            self.model = model
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.name_or_path)
        if init_on_cuda:
            self.cuda()
        self.dataset = ArrowList(dataset.to_dict()) if dataset else None
        self.features = features
        if self.dataset:
            self.faiss_index = self.build_faiss_index(dataset)
            self.features = features or dataset.features

    def cuda(self, device=None):
        # convenience: move all modeling components as if this was a nn.Module
        self.model = self.model.cuda(device)

    def to(self, *args, **kwargs):
        # convenience: move all modeling components as if this was a nn.Module
        self.model = self.model.to(*args, **kwargs)

    def encode_turn(self, turn: DatasetTurn, to_np_array: bool = False) -> Union[torch.Tensor, npt.NDArray]:
        # start simple: only the most recent user utterance
        # tokenize -> move tensors to model device -> encode with model -> get pooled output for single utterance
        with torch.no_grad():
            embedding: torch.Tensor = self.model(**{
                k: v.to(self.model.device) for k, v in
                self.tokenizer(self.turn_encoding_context_fn(turn), return_tensors="pt").items()
            })['pooled_output'][0]

            # return a numpy array if we asked for it
            return embedding.cpu().numpy() if to_np_array else embedding

    def encode_turn_batch(self, turn_batch: HasEncoderInput, to_np_array: bool = False) -> Union[
        torch.Tensor, npt.NDArray]:
        # tokenize -> move tensors to model device -> encode with model -> get pooled output for single utterance
        with torch.no_grad():
            embedding: torch.Tensor = self.model(**{
                k: v.to(self.model.device) for k, v in
                self.tokenizer(turn_batch["encoder_input"], return_tensors="pt", padding=True, truncation=True).items()
            })['pooled_output']

            # return a numpy array if we asked for it
            return embedding.cpu().numpy() if to_np_array else embedding

    def get_nearest_examples(self, turn: DatasetTurn, k: int = 10) -> List[DatasetTurn]:
        """
        Return the k nearest turns in the dataset according to the dense retriever (top-k decoding). If k is greater
        than the number of turns in the dataset, returning the complete dataset.

        :param turn: turn to use as query
        :param k: number of nearest turns to return
        :return: nearest turns
        """
        if not self.dataset:
            logging.warning("retrieving from a module that has no items")
            return []
        if k > len(self.dataset):
            # skip retrieval and return the whole thing
            logging.warning(f"requesting k={k} examples from a dataset of only lenggth={len(self.dataset)}")
            return [t for t in self.dataset]
        query: npt.NDArray = self.encode_turn(turn, to_np_array=True)
        results: SearchResults = self.faiss_index.search(query=query, k=k)
        return [t for t in self.dataset.select(results.indices)]

    def get_nearest_examples_batched(self, turns: List[DatasetTurn], k: int = 10) -> List[List[DatasetTurn]]:
        """
        Return the k nearest turns in the dataset according to the dense retriever (top-k decoding). If k is greater
        than the number of turns in the dataset, returning the complete dataset.

        :param turn: turn to use as query
        :param k: number of nearest turns to return
        :return: nearest turns
        """
        if not self.dataset:
            logging.warning("retrieving from a module that has no items")
            return [[] for _ in turns]
        if k > len(self.dataset):
            # skip retrieval and return the whole thing
            logging.warning(f"requesting k={k} examples from a dataset of only length={len(self.dataset)}")
            return [[t for t in self.dataset] for _ in turns]
        turn_inputs: HasEncoderInput = {"encoder_input": [self.turn_encoding_context_fn(turn) for turn in turns]}
        queries: npt.NDArray = self.encode_turn_batch(turn_inputs, to_np_array=True)
        scores, indices = self.faiss_index.search_batch(queries=queries, k=k)
        result: List[List[DatasetTurn]] = []
        for batch_i in range(len(turns)):
            result.append([self.dataset[j] for j in indices[batch_i]])
        return result


        return [t for t in self.dataset.select(results.indices)]

    def get_nearest_indices_batch(self, turn_batch: HasEncoderInput, k: int = 10) -> npt.NDArray:
        return self.get_nearest_batch(turn_batch, k)[1]

    def get_nearest_batch(self, turn_batch: HasEncoderInput, k: int = 10) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        This returns all data points in the nearest batch. Note: they are returned a flat sequence of k entries per
        batch item. So the nearest to turn_batch[0] = result[0], turn_batch[1] = result[k], turn_batch[2] = result[2*k],
        etc.
        """
        batch_query: Tensor = self.encode_turn_batch(turn_batch, to_np_array=True)
        scores, indices = self.faiss_index.search_batch(batch_query, k)
        return scores, indices

    def add_items(self, turns: Union[List[DatasetTurn], Dataset]) -> None:
        """
        Adds items to the dataset and retrieval index. This is NOT idempotent and does NOT check for duplicates,
        so call carefully

        :param turns: turns to add to the dataset
        :return: None
        """
        if not self.dataset:
            # lazily instantiating our dataset and index
            dataset = Dataset.from_list(turns, features=self.features)
            self.dataset = ArrowList(dataset.to_dict())
            self.faiss_index = self.build_faiss_index(dataset=dataset)
        else:
            # updating an existing index and dataset
            if type(turns) == Dataset:
                self.dataset.append(turns.to_dict())
            elif turns:
                self.dataset.append({k: [turn[k] for turn in turns] for k in turns[0]})
            turn_batch: HasEncoderInput = {
                "encoder_input": [self.turn_encoding_context_fn(turn) for turn in turns]
            }
            vectors: npt.NDArray = self.encode_turn_batch(turn_batch, to_np_array=True)
            self.faiss_index.add_vectors(vectors)

    def build_faiss_index(self, dataset: Dataset, device: int = None) -> FaissIndex:
        faiss_index: FaissIndex = FaissIndex(device=device, metric_type=faiss.METRIC_INNER_PRODUCT)

        input_dataset: Dataset = dataset.map(map_to_encoder_input, remove_columns=dataset.column_names,
                                             num_proc=min(cpu_count(), 32))
        embedded_dataset: Dataset = input_dataset.map(lambda turn_batch: {
            "embedding": self.encode_turn_batch(turn_batch, to_np_array=True)
        }, batched=True, batch_size=512)
        embedded_dataset.set_format('numpy', columns=['embedding'])
        faiss_index.add_vectors(
            embedded_dataset, column="embedding",
            train_size=len(embedded_dataset) if self.train_index else None
        )
        return faiss_index

    def load_faiss_index(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset) if self.dataset else 0
