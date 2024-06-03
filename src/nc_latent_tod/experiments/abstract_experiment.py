import abc
import copy
import logging
from typing import Optional, Callable, List, Dict

from datasets import Dataset

from nc_latent_tod.acts.act import Act
from nc_latent_tod.data_types import DatasetTurnLog
from nc_latent_tod.delex.abstract_delexer import AbstractDelexer
from nc_latent_tod.delex.fuzzy_regex_delexer import FuzzyRegexDelexer
from nc_latent_tod.experiments.data_types import RGEvaluationInput
from nc_latent_tod.acts.utils import get_acts_from_system_acts
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.normalization.schema_normalizer import SchemaNormalizer
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.utils.general import group_by_dial_id_and_turn
from nc_latent_tod.utils.turn_logger import TurnLogger
from nc_latent_tod.utils.wandb_step_logger import WandbStepLogger


class AbstractExperiment(metaclass=abc.ABCMeta):
    """
    Abstract class for all experiment types, covering basics like output directories, etc.
    """

    test_set: Dataset
    train_set: Dataset
    output_dir: str
    logger: WandbStepLogger
    turn_logger: TurnLogger
    schema: List[ServiceSchema]
    batch_size: int

    def __init__(self, *, test_set: Dataset, train_set: Optional[Dataset] = None, output_dir: str = None,
                 schema_loader: Callable[[], List[ServiceSchema]] = None, batch_size: int = 1,
                 delexer: AbstractDelexer = None, normalizer: AbstractNormalizer = None, **kwargs) -> None:
        self.test_set = test_set
        self.train_set = train_set
        self.output_dir = output_dir
        self.schema = schema_loader()
        self.logger = WandbStepLogger()
        self.turn_logger = TurnLogger()
        self.batch_size = batch_size
        if not delexer:
            logging.warning('No delexer provided, using default FuzzyRegexDelexer')
        self.delexer = delexer or FuzzyRegexDelexer()
        if not normalizer:
            logging.warning('No normalizer provided, using default SchemaNormalizer')
        self.normalizer = normalizer or SchemaNormalizer(self.schema)

    def build_rg_evaluation_input(self, logs: List[DatasetTurnLog]) -> RGEvaluationInput:
        # prepare inputs for mwzeval: set up dictionary
        eval_input: RGEvaluationInput = {}

        # group logs by dialogue id and turn id
        logs_by_dial_id_and_turn: Dict[str, List[DatasetTurnLog]] = group_by_dial_id_and_turn(
            logs, dialogue_id_key='dialogue_id', turn_id_key='turn_id'
        )
        for dialogue_id, turns in logs_by_dial_id_and_turn.items():
            eval_dialogue_id: str = dialogue_id.lower().replace('.json', '')
            eval_input[eval_dialogue_id] = []
            for log in turns:
                # for each turn, get the system acts
                acts: List[Act] = get_acts_from_system_acts(log['pred_system_response_acts'], self.schema)

                # re-delexify the response: if we predict a response like "Curriz and Grills serves [value_food] food",
                # then we want to capture the name if we can on re-delex, assuming the name is in the act
                re_delex_response: str = self.delexer.delexify(log['pred_delex_system_response'], acts,
                                                               transform_slots_for_eval=True)
                eval_input[eval_dialogue_id].append({
                    "dialogue_id": dialogue_id,
                    "turn_id": log['turn_id'],
                    "response": re_delex_response,
                    "state": copy.deepcopy(self.normalizer.normalize(log['pred_belief_state'])),
                    "active_domains": list(
                        set(log['pred_belief_state'].keys()).union(log['pred_act_based_active_service_names'])
                    )
                })
        return eval_input
