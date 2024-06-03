import logging
from typing import Union, List, Optional, Callable

import faiss
import numpy.typing as npt
import torch
from datasets import Dataset, Features
from datasets.search import FaissIndex, SearchResults
from nc_latent_tod.retrieval.mpnet.mpnet_model_for_retrieval import MPNetModelForRetrieval
from nc_latent_tod.retrieval.mpnet.mpnet_retriever import get_context_system_user_encoding, HasEncoderInput, \
    MPNetRetriever
from nc_latent_tod.utils.arrow_list import ArrowList
from torch import Tensor
from transformers import AutoTokenizer

from nc_latent_tod.data_types import DatasetTurn


class ContaminantMPNetRetriever(MPNetRetriever):
    """
    Composes the model and tokenizer in order to facilitate contaminated document retrieval
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

    def __init__(self, model: MPNetModelForRetrieval = None, tokenizer: AutoTokenizer = None, dataset: Dataset = None,
                 features: Features = None, model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2",
                 train_index: bool = False, init_on_cuda: bool = False,
                 turn_encoding_context_fn: Callable[[DatasetTurn], str] = get_context_system_user_encoding,
                 documents: List[str] = None,
                 **kwargs) -> None:
        default_features = Features({"content": {"type": "string"}})
        super().__init__(model, tokenizer, dataset, features or default_features, model_name_or_path, train_index, init_on_cuda,
                         turn_encoding_context_fn, **kwargs)
        self.documents = documents
        self.document_index = self.build_document_faiss_index(documents)

    def encode_document(self, content: str, to_np_array: bool = False) -> Union[torch.Tensor, npt.NDArray]:
        # tokenize -> move tensors to model device -> encode with model -> get pooled output for single utterance
        with torch.no_grad():
            embedding: torch.Tensor = self.model(**{
                k: v.to(self.model.device) for k, v in
                self.tokenizer(content, return_tensors="pt").items()
            })['pooled_output'][0]

            # return a numpy array if we asked for it
            return embedding.cpu().numpy() if to_np_array else embedding

    def encode_document_batch(self, contents: List[str], to_np_array: bool = False) -> Union[
        torch.Tensor, npt.NDArray]:
        # tokenize -> move tensors to model device -> encode with model -> get pooled output for single utterance
        with torch.no_grad():
            embedding: torch.Tensor = self.model(**{
                k: v.to(self.model.device) for k, v in
                self.tokenizer(contents, return_tensors="pt", padding=True, truncation=True).items()
            })['pooled_output']

            # return a numpy array if we asked for it
            return embedding.cpu().numpy() if to_np_array else embedding

    def get_nearest_documents(self, turn: DatasetTurn, k: int = 10) -> List[str]:
        """
        Return the k nearest documents in the dataset according to the dense retriever (top-k decoding). If k is greater
        than the number of turns in the dataset, returning the complete dataset.

        :param turn: turn to use as query
        :param k: number of nearest turns to return
        :return: nearest turns
        """
        if not self.documents:
            logging.warning("retrieving from a module that has no items")
            return []
        if k > len(self.documents):
            # skip retrieval and return the whole thing
            logging.warning(f"requesting k={k} examples from a dataset of only length={len(self.documents)}")
            return [t for t in self.documents]
        query: npt.NDArray = self.encode_turn(turn, to_np_array=True)
        results: SearchResults = self.document_index.search(query=query, k=k)
        return [self.documents[i] for i in results.indices]

    def get_nearest_examples_batched(self, turns: List[DatasetTurn], k: int = 10) -> List[List[str]]:
        """
        Return the k nearest turns in the dataset according to the dense retriever (top-k decoding). If k is greater
        than the number of turns in the dataset, returning the complete dataset.

        :param turn: turn to use as query
        :param k: number of nearest turns to return
        :return: nearest turns
        """
        if not self.documents:
            logging.warning("retrieving from a module that has no items")
            return [[] for _ in turns]
        if k > len(self.documents):
            # skip retrieval and return the whole thing
            logging.warning(f"requesting k={k} examples from a dataset of only length={len(self.documents)}")
            return [[t for t in self.documents] for _ in turns]
        turn_inputs: HasEncoderInput = {"encoder_input": [self.turn_encoding_context_fn(turn) for turn in turns]}
        queries: npt.NDArray = self.encode_turn_batch(turn_inputs, to_np_array=True)
        scores, indices = self.document_index.search_batch(queries=queries, k=k)
        result: List[List[str]] = []
        for batch_i in range(len(turns)):
            result.append([self.documents[j] for j in indices[batch_i]])
        return result

    def build_document_faiss_index(self, documents: List[str], device: int = None) -> FaissIndex:
        faiss_index: FaissIndex = FaissIndex(device=device, metric_type=faiss.METRIC_INNER_PRODUCT)

        # iterate over documents in groups of 32:
        for i in range(0, len(documents), 32):
            batch: List[str] = documents[i:i + 32]
            batch_embedding: Tensor = self.encode_document_batch(batch)
            faiss_index.add_vectors(
                batch_embedding, train_size=len(batch) if self.train_index else None
            )
        return faiss_index

    def load_faiss_index(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.documents) if self.documents else 0


if __name__ == '__main__':
    query_turn: DatasetTurn = {
        "user_utterances": ["I would like to know the weather in Paris"],
        "system_utterances": ["Sure, I can help you with that. The weather in Paris is currently 20 degrees Celsius"],
        "last_slot_values": {},
        "slot_values": {},
        "system_response": "The weather in Paris is currently 20 degrees Celsius",
        "turn_id": 0,
        "dialogue_id": "1234",
        "turn_slot_values": {},
        "system_response_acts": [],
        "last_system_response_acts": [],
        "domains": [],
    }
    documents: List[str] = [
        "something about the weather in Paris is great",
        "some irrelevant document",
        "I would like to know the weather in Paris",
        "is there food on Mars"
    ]

    retriever: ContaminantMPNetRetriever = ContaminantMPNetRetriever(
        documents=documents,
    )
    print(retriever.get_nearest_documents(query_turn, k=2))

