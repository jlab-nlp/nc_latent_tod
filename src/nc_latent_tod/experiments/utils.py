import copy
import dataclasses
import logging
import os
import random
from collections import defaultdict
from typing import Callable, List, Generator, Dict, Tuple, Optional, Sequence, Type, Any, TypeVar, Set

import dacite
import torch
import wandb
from datasets import Dataset, load_dataset
from nc_latent_tod.retrieval.mpnet.distinct_mpnet_retriever import DistinctMPNetRetriever
from nc_latent_tod.retrieval.random_retriever import RandomRetriever
from transformers import PreTrainedModel

from nc_latent_tod.acts.utils import get_acts_slots_as_key_string
from nc_latent_tod.clients.abstract_hf_lm_client import AbstractHFLMClient
from nc_latent_tod.clients.abstract_lm_client import AbstractLMClient
from nc_latent_tod.clients.codex_client import CodexClient
from nc_latent_tod.clients.starcoder_client import StarCoderClient
from nc_latent_tod.data.utils import get_hf_dataset_features
from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.db.abstract_db import AbstractDB
from nc_latent_tod.db.multiwoz_db import MultiWOZDB
from nc_latent_tod.experiments.batch_client_lm_module import AbstractLMClientModule, BatchLMClientDSTModule, \
    BatchLMClientActTagModule, BatchLMClientPolicyModule, BatchLMClientResponseGenModule
from nc_latent_tod.experiments.config import DataConfigDC, LMModelConfig, ExampleRetrieverConfigDC, LMExperimentConfig, \
    BaseExperimentConfig, BatchModuleConfig, RetrievalInputFormat, SchemaConfigDC
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator
from nc_latent_tod.retrieval.abstract_retriever import AbstractRetriever
from nc_latent_tod.retrieval.mpnet.mpnet_retriever import MPNetRetriever, get_context_system_user_encoding
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.reader import read_multiwoz_schema
from nc_latent_tod.utils.artifacts import output_dir_to_run_or_artifact_name
from nc_latent_tod.utils.dialogue_states import get_state_slots_as_key_string
from nc_latent_tod.utils.general import get_project_output_root_dir, read_json, NC_LATENT_TOD_OUTPUTS_DIR

T = TypeVar('T', bound=BaseExperimentConfig)


def read_experiment_config(config_path: str, data_class: Type[T] = LMExperimentConfig,
                           run_id: str = None) -> T:
    """
    Reads the config as a data_class given by Type using dacite, in strict mode.
    """
    config_data: Dict[str, Any] = read_json(config_path)
    return parse_experiment_config(config_data, config_path, data_class, run_id=run_id)


def parse_experiment_config(config_data: Any, config_path: str, data_class: Type[T] = BaseExperimentConfig,
                            run_id: str = None) -> T:
    # no generic way to do this, so we'll just do it manually: need to clear instances of DatasetTurn since strict mode
    # type checking will throw an error if a Type signature is a sub-class of TypedDict. We'll just remove the examples,
    # since they are an `Optional[List[DatasetTurn]]`. Then, load after the fact, trusting our value implicitly.
    may_have_examples: List[str] = ['model', 'dst_model', 'act_model']
    examples: Dict[str, List[DatasetTurn]] = {}
    for config_key in may_have_examples:
        if config_key in config_data:
            if 'examples' in config_data[config_key]:
                examples[config_key] = config_data[config_key]['examples']
                config_data[config_key]['examples'] = None

    cfg: T = dacite.from_dict(
        data_class=data_class,
        data=config_data,
        config=dacite.Config(strict=True, cast=[])
    )
    for key, example_values in examples.items():
        has_examples: Any = getattr(cfg, key)
        # set the attribute directly, since we can't check the type of a TypedDict in dacite.from_dict
        has_examples.examples = example_values

    # set some other reasonable defaults, like wandb run name, and output_dir (move if relative path, to where rooted
    # in the data dir
    default_output_dir: str = os.path.join(os.path.dirname(config_path), run_id) if run_id else os.path.dirname(
        config_path)
    output_dir: str = cfg.output.output_dir or default_output_dir
    full_output_dir: str = os.path.join(get_project_output_root_dir(), output_dir)
    cfg.output.output_dir = full_output_dir
    default_run_name: str = output_dir_to_run_or_artifact_name(config_path)
    default_run_group: str = default_run_name.rsplit('-', maxsplit=1)[0]
    cfg.wandb.run_name = cfg.wandb.run_name or default_run_name
    cfg.wandb.run_group = cfg.wandb.run_group or default_run_group
    return cfg


def get_schema_loader_by_type(schema_name_or_type: str) -> Callable[[], List[ServiceSchema]]:
    if schema_name_or_type == "multiwoz":
        return read_multiwoz_schema
    else:
        raise ValueError(f"unknown schema {schema_name_or_type}")


def get_turn_ordered_batch_iterator(dataset: Dataset, batch_size: int = 1, num_partitions: int = 0, warmup: int = 0,
                                    skip_turns_in: List[DatasetTurn] = None) -> Generator[List[DatasetTurn], None, None]:
    if skip_turns_in:
        skip_dial_ids_and_turn_ids: Set[Tuple[str, int]] = set(
            (turn['dialogue_id'], turn['turn_id']) for turn in skip_turns_in
        )
        dataset = dataset.filter(lambda turn: (turn['dialogue_id'], turn['turn_id']) not in skip_dial_ids_and_turn_ids,
                                 desc="filtering out turns in skip_turns_in")
    partitions: List[Dataset] = []
    total_yielded: int = 0
    logging.info(f"Creating iterator with batch_size={batch_size}, num_partitions={num_partitions}, warmup={warmup}")
    if num_partitions > 0:
        dialogue_ids: List[str] = dataset.unique('dialogue_id')
        local_random = random.Random(42)
        local_random.shuffle(dialogue_ids)
        partition_size: int = len(dialogue_ids) // num_partitions
        for i in range(num_partitions):
            start: int = i * partition_size
            # make sure to be full at the end:
            end: int = (i + 1) * partition_size
            if i == num_partitions - 1:
                end = len(dialogue_ids)
            partition_dialogue_ids: Set[str] = set(dialogue_ids[start:end])
            partitions.append(dataset.filter(lambda turn: turn['dialogue_id'] in partition_dialogue_ids))
    else:
        # all in one partition
        partitions.append(dataset)
    for partition in partitions:
        turn_ids: List[int] = sorted(partition.unique('turn_id'))
        for turn_id in turn_ids:
            subset = partition.filter(lambda turn: turn['turn_id'] == turn_id)
            # don't do any fancy collating, just give lists as-is
            for i in range(0, len(subset), batch_size):
                batch = subset.select(range(i, min(i + batch_size, len(subset))))
                if warmup > 0 and total_yielded < warmup:
                    # yield batches of size 1 for warmup
                    for turn in batch:
                        yield [turn]
                    total_yielded += len(batch)
                else:
                    yield batch
                    total_yielded += len(batch)


def load_datasets_from_cfg(data_cfg: DataConfigDC) -> Tuple[Dataset, Dataset]:
    """
    Returns the training and evaluation dataset as configured in the data config, in
    that order as a tuple. None will be returned for non-configured datasets
    """
    train_set, eval_set = None, None
    if data_cfg.train_set_path_or_name:
        train_set = load_dataset(
            data_cfg.train_set_path_or_name,
            split=data_cfg.train_set_split_name or 'train',
            token=True
        )
    if data_cfg.eval_set_path_or_name:
        eval_set = load_dataset(
            data_cfg.eval_set_path_or_name,
            split=data_cfg.eval_set_split_name or 'validation',
            token=True
        )
    return train_set, eval_set


def get_client_from_lm_model_config(lm_model_cfg: LMModelConfig, **kwargs) -> AbstractLMClient:
    """
    Given a configuration for an LM client from a model config, construct and return the client for that LM.
    """
    for key, value in kwargs.items():
        if hasattr(lm_model_cfg, key):
            setattr(lm_model_cfg, key, value)

    torch_dtype = None
    if lm_model_cfg.torch_dtype:
        torch_dtype = getattr(torch, lm_model_cfg.torch_dtype) if lm_model_cfg.torch_dtype != 'auto' else 'auto'
    client_kwargs = {
        "stop_sequences": lm_model_cfg.stop_sequences,
        "model_name_or_path": lm_model_cfg.model_name_or_path,
        "adapter_path": lm_model_cfg.adapter_path,
        "max_batch_size": lm_model_cfg.batch_size,
        "load_in_8bit": lm_model_cfg.load_in_8bit,
        "attn_implementation": lm_model_cfg.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_past_key_value_cache": lm_model_cfg.use_past_key_value_cache,
        **kwargs
    }
    if lm_model_cfg.model_type == "code-davinci-002":
        return CodexClient(stop_sequences=lm_model_cfg.stop_sequences)
    elif lm_model_cfg.model_type == "starcoder":
        return StarCoderClient(**client_kwargs, torch_compile=False)
    else:
        raise ValueError(f"unsupported LM model type: {lm_model_cfg.model_type}")


T = TypeVar('T', bound=AbstractLMClientModule)


def get_db_from_schema_config(schema_config: SchemaConfigDC) -> AbstractDB:
    if schema_config.name == "multiwoz":
        return MultiWOZDB()
    else:
        raise ValueError(f"unknown schema {schema_config.name}")


def get_item_distinguishing_fn_by_module_type(module_type: str) -> Callable[[DatasetTurn], Any]:
    return {
        "dst": lambda turn: get_state_slots_as_key_string(turn['turn_slot_values']),
        "act_tag": lambda turn: get_acts_slots_as_key_string(turn['system_response_acts']),
        # these aren't used, but hypothetically we'd probably want our in-context examples for policy/RG to reflect
        # distinct system actions
        "policy": lambda turn: get_acts_slots_as_key_string(turn['system_response_acts']),
        "response_gen": lambda turn: get_acts_slots_as_key_string(turn['system_response_acts']),
    }[module_type]


class ModelCachingModuleBuilder:

    cache: Dict[str, Dict[str, PreTrainedModel]]

    def __init__(self):
        # model_type -> model_name_or_path -> model
        self.cache = defaultdict(dict)

    def get_cached_model(self, lm_model_config: LMModelConfig) -> Optional[PreTrainedModel]:
        if lm_model_config.adapter_path:
            # not supporting caching of adapted models
            return None
        if lm_model_config.do_not_cache:
            return None
        if lm_model_config.model_type == 'text_gen_inference':
            return None
        return self.cache[lm_model_config.model_type].get(lm_model_config.model_name_or_path)

    def cache_model(self, lm_model_config: LMModelConfig, model: PreTrainedModel):
        if lm_model_config.adapter_path:
            # not supporting caching of adapted models
            return
        if lm_model_config.do_not_cache:
            return
        self.cache[lm_model_config.model_type][lm_model_config.model_name_or_path] = model

    def build_batch_module_from_config(self, cfg: BatchModuleConfig, prompt_generator: AbstractPromptGenerator, **module_kwargs) -> T:
        class_type = {
            "dst": BatchLMClientDSTModule,
            "act_tag": BatchLMClientActTagModule,
            "policy": BatchLMClientPolicyModule,
            "response_gen": BatchLMClientResponseGenModule
        }[cfg.module_type]

        item_distinguishing_fn: Callable[[DatasetTurn], Any] = get_item_distinguishing_fn_by_module_type(cfg.module_type)

        retriever_config: Optional[ExampleRetrieverConfigDC] = cfg.retriever
        model_config: Optional[LMModelConfig] = cfg.model
        retriever: Optional[AbstractRetriever] = None
        if retriever_config.use_retriever:
            retriever = build_retriever_from_config(
                retriever_config,
                schema=prompt_generator.schema,
                item_distinguishing_fn=item_distinguishing_fn
            )
        cached_model: Optional[PreTrainedModel] = self.get_cached_model(model_config)
        if cached_model:
            client: AbstractLMClient = get_client_from_lm_model_config(model_config, model=cached_model)
        else:
            client: AbstractLMClient = get_client_from_lm_model_config(model_config)
            if isinstance(client, AbstractHFLMClient):
                self.cache_model(model_config, client.model)
        verbatim_documents: Optional[List[str]] = None
        if cfg.verbatim_k_examples > 0 and cfg.verbatim_documents_path:
            verbatim_documents = read_json(cfg.verbatim_documents_path)
        return class_type(
            prompt_generator=prompt_generator,
            client=client,
            examples=model_config.examples,
            retriever=retriever,
            retrieve_k_examples=retriever_config.k_examples,
            example_warmup=retriever_config.example_warmup,
            retriever_add_to_index=cfg.add_predictions_to_index,
            generation_cfg=cfg.generation_cfg,
            verbatim_k_examples=cfg.verbatim_k_examples,
            verbatim_contaminants=verbatim_documents,
            **module_kwargs
        )


def build_retriever_from_config(retriever_cfg: ExampleRetrieverConfigDC,
                                schema: Optional[List[ServiceSchema]] = None,
                                item_distinguishing_fn: Callable[[DatasetTurn], Any] = repr) -> AbstractRetriever:
    """
    Given a configuration for an in-context turn retriever, construct and return that retriever.
    """
    retrieval_input_format: RetrievalInputFormat = retriever_cfg.retrieval_input_format or "context_system_user"
    if retrieval_input_format == "context_system_user":
        turn_encoding_context_fn = get_context_system_user_encoding
    elif retrieval_input_format == "response":
        def get_response(turn) -> str:
            return turn['system_response']

        turn_encoding_context_fn = get_response
    elif retrieval_input_format == "user_response":
        def get_user_response(turn) -> str:
            return f"User: {turn['user_utterances'][-1]}\nSystem: {turn['system_response']}"

        turn_encoding_context_fn = get_user_response
    else:
        raise ValueError(f"unsupported retrieval input format: {retrieval_input_format}")
    retriever_cfg = copy.deepcopy(retriever_cfg)  # don't modify the original
    index_dataset: Optional[Sequence[DatasetTurn]] = []
    if retriever_cfg.index_set_path_or_name:
        index_dataset = load_dataset(retriever_cfg.index_set_path_or_name,
                                     split=retriever_cfg.index_set_split_name)
    if retriever_cfg.type == 'mpnet':
        if retriever_cfg.from_artifact:
            # download the artifact and change the model name or path to its location
            artifact = wandb.use_artifact(retriever_cfg.model_name_or_path)
            retriever_cfg.model_name_or_path = artifact.download()
        assert schema is not None, "schema must be provided to build MPNetRetriever"
        additional_kwargs = {}
        retriever_class: Type[AbstractRetriever] = MPNetRetriever
        if retriever_cfg.minimum_distinct > 1:
            retriever_class = DistinctMPNetRetriever
            additional_kwargs['distinguishing_fn'] = item_distinguishing_fn

        return retriever_class(
            features=get_hf_dataset_features(schema),
            dataset=index_dataset,
            # We'll assume if we're indexing a dataset, then it will be worthwhile to train it
            train_index=index_dataset is not None,
            init_on_cuda=torch.cuda.is_available(),
            turn_encoding_context_fn=turn_encoding_context_fn,
            **additional_kwargs,
            **dataclasses.asdict(retriever_cfg)
        )
    elif retriever_cfg.type == 'random':
        return RandomRetriever(pool=index_dataset)
    else:
        raise ValueError(f"unsupported retriever type: {retriever_cfg.type}")


def verify_cuda_is_available_if_needed():
    if os.environ.get("REQUIRE_CUDA", 0):
        assert torch.cuda.is_available(), "CUDA is required for this experiment, but it is not available. " \
                                          "CUDA_VISIBLE_DEVICES: " + os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")


def scale_batch_size_to_memory(batch_size: int) -> int:
    # we configure batch sizes that make sense for an A100, but want to be able to seemlessly use A40s, etc.
    if not torch.cuda.is_available():
        # trust that we are in an environment where we don't need to scale (API), not that we're running on CPU
        return batch_size
    memory_in_gb: float = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if memory_in_gb < 25:
        new_batch_size = 1
    elif memory_in_gb < 50:
        new_batch_size = batch_size // 2
    else:
        new_batch_size = batch_size
    new_batch_size = max(1, new_batch_size)
    return new_batch_size
