from dataclasses import dataclass, field
from typing import Optional, Callable, List, Literal, Any, Dict

from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.ontology.abstract_ontology import AbstractDBOntology
from nc_latent_tod.ontology.multiwoz.ontology import MultiWOZOntology
from nc_latent_tod.prompting.abstract_prompt_generator import PromptMode
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.reader import read_multiwoz_schema


@dataclass
class OutputConfigDC:
    output_dir: str = None
    on_end_copy_to_dir: str = None


@dataclass
class WandbConfigDC:
    run_name: str = None
    run_group: str = None
    run_notes: str = None
    run_tags: Optional[List[str]] = None


@dataclass
class DatasetConfig:
    """
    Configuration for a single dataset split that is required. An example: the output of a self-labelling process
    should specify an exact path_or_name and split to save the labelled dataset to.
    """
    path_or_name: str
    split_name: str
    push_to_hub: bool = False


@dataclass
class DataConfigDC:
    """
    Input data configuration for an experiment. Since different experiments may or may not have train/eval data, both
    are marked optional.
    """
    train_set_path_or_name: Optional[str]
    train_set_split_name: Optional[str]
    eval_set_path_or_name: Optional[str]
    eval_set_split_name: Optional[str]


@dataclass
class BaseExperimentConfig:
    data: DataConfigDC
    # noinspection PyArgumentList
    output: OutputConfigDC = OutputConfigDC()
    # noinspection PyArgumentList
    wandb: WandbConfigDC = WandbConfigDC()
    profile: bool = False  # deprecated: flag was used for line_profiler, but is not used anymore


RetrieverType = Literal["mpnet", "queue", "random"]
RetrievalInputFormat = Literal["context_system_user", "user_response", "response"]

@dataclass
class ExampleRetrieverConfigDC:
    """
    Configuration for an in-context example retriever

    Fields:
        use_retriever: whether to use the retriever (default is False)
        type: retriever type (see RetrieverType for supported types, default is 'mpnet')
        model_name_or_path: path to the retriever model (default is None)
        k_examples: number of examples to retrieve (default is 0)
        index_set_path_or_name: path to the index dataset to retrieve from (default is None)
        index_set_split_name: name of the split to retrieve from (default is None)
        add_to_index: whether to add predictions on new examples to the retriever index (default is False)
        from_artifact: whether to load the retriever from an artifact, specified from model_name_or_path (default=False)
    """
    use_retriever: bool = False
    type: Optional[RetrieverType] = "mpnet"
    model_name_or_path: str = None
    k_examples: int = 0
    example_warmup: int = 0
    index_set_path_or_name: str = None
    index_set_split_name: str = None
    from_artifact: bool = False
    retrieval_input_format: Optional[RetrievalInputFormat] = None
    use_normalizer: bool = False
    minimum_distinct: Optional[int] = 1


@dataclass
class SchemaConfigDC:
    """
    loading schema and related elements
    """
    schema_loader: Optional[Callable[[], List[ServiceSchema]]]
    ontology: Optional[Callable[[], AbstractDBOntology]]
    ontology_file: Optional[str]
    name: str = "multiwoz"

# Some reasonable defaults for using MultiWOZ
MultiWOZSchemaConfig = SchemaConfigDC(
    name='multiwoz',
    schema_loader=read_multiwoz_schema,
    ontology=MultiWOZOntology.create_ontology,
    ontology_file=f"db/multiwoz/2.4/ontology.json"
)

FullMultiWOZSchemaConfig = SchemaConfigDC(
    name='multiwoz',
    schema_loader=lambda: read_multiwoz_schema(only_evaluated_schemas=False),
    ontology=MultiWOZOntology.create_ontology,
    ontology_file=f"db/multiwoz/2.4/ontology.json"
)


LMClientType = Literal["code-davinci-002", "starcoder", "code_llama", "mock_simple"]


@dataclass
class LMModelConfig:
    stop_sequences: List[str]
    examples: Optional[List[DatasetTurn]] = None
    model_name_or_path: str = None
    batch_size: int = 1
    adapter_path: str = None
    model_type: LMClientType = "code-davinci-002"
    load_in_8bit: bool = True
    do_not_cache: bool = False
    attn_implementation: Optional[str] = None
    torch_dtype: Optional[str] = None
    use_past_key_value_cache: bool = False
    # only used when model_type == 'mock_perfect', used for perfect predictions or for repeating from a W&B run
    mock_from_run_id: Optional[str] = None


@dataclass
class LMExperimentConfig(BaseExperimentConfig):
    model: LMModelConfig = LMModelConfig(model_type=None, stop_sequences=[])
    retriever: ExampleRetrieverConfigDC = ExampleRetrieverConfigDC()
    create_self_labeled_dataset: bool = False
    add_predictions_to_index: bool = False
    schema_config: SchemaConfigDC = MultiWOZSchemaConfig
    prompt_mode: PromptMode = None


ModuleType = Literal["dst", "policy", "act_tag", "response_gen"]

# Defining a literal for common generation modes, e.g. 'greedy' and 'noisy_channel_joint'
GenerationMode = Literal['greedy', 'noisy_channel_joint', 'noisy_channel_cond']


@dataclass
class GenerationConfig:
    prompt_mode: PromptMode = None
    generation_mode: GenerationMode = "greedy"
    noisy_channel_prompt_mode: PromptMode = None
    sampling_args: Dict[str, Any] = field(default_factory=lambda: {
        "top_p": 0.9,
        "n": 5,
        "best_of": 10
    })


@dataclass
class BatchModuleConfig:
    module_type: ModuleType = None
    model: LMModelConfig = LMModelConfig(model_type=None, stop_sequences=[])
    retriever: ExampleRetrieverConfigDC = ExampleRetrieverConfigDC()
    generation_cfg: GenerationConfig = GenerationConfig()
    add_predictions_to_index: bool = False
    verbatim_k_examples: int = 0
    verbatim_documents_path: Optional[str] = None


@dataclass
class ManifestConfig:
    type: Literal["dynamo", "local"] = "local"
    manifest_path: str = None
    group_id: str = None
    write_to_manifest: bool = False
    seed_retrievers_from_manifest: bool = False
    manifest_must_exist: bool = False  # if true, will fail if manifest does not exist
    group_id_must_exist: bool = False  # if true, will fail if group_id does not exist in manifest


PGType = Literal["kwargs", "simple_kwargs"]


@dataclass
class OfflineLabellingLMExperimentConfig(BaseExperimentConfig):
    dst: BatchModuleConfig = BatchModuleConfig()
    act_tag: BatchModuleConfig = BatchModuleConfig()
    create_self_labelled_dataset: bool = False
    schema_config: SchemaConfigDC = MultiWOZSchemaConfig
    publish_labelled_dataset_as: Optional[DatasetConfig] = None
    data_num_partitions: int = 0
    data_warmup: int = 0
    resume_from_logs: Optional[str] = None
    manifest: ManifestConfig = ManifestConfig()
    prompt_generator: Optional[PGType] = "kwargs"


@dataclass
class OnlineE2EEvalLMExperimentConfig(BaseExperimentConfig):
    dst: BatchModuleConfig = BatchModuleConfig()
    policy: BatchModuleConfig = BatchModuleConfig()
    response_gen: BatchModuleConfig = BatchModuleConfig()
    create_self_labelled_dataset: bool = False
    schema_config: SchemaConfigDC = MultiWOZSchemaConfig
    publish_labelled_dataset_as: Optional[DatasetConfig] = None
    prompt_generator: Optional[PGType] = "kwargs"
