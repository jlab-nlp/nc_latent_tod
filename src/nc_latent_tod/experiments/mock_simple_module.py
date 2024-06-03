from typing import Any, List, Dict, Tuple, Literal

from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.experiments.batch_client_lm_module import AbstractLMClientModule
from nc_latent_tod.experiments.batch_client_lm_module import TaskInput, TaskOutput
from nc_latent_tod.experiments.data_types import GenericOutputs, SchemaGuidedDSTOutputs, SchemaGuidedActTaggingOutputs


OutputType = Literal["dst", "act_tagging"]


class SimpleMockModule(AbstractLMClientModule):

    def get_task_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> str:
        pass

    def get_noisy_channel_prompt(self, task_input: TaskInput, examples: List[DatasetTurn]) -> str:
        pass

    def get_noisy_channel_completion(self, task_input: TaskInput, completion: str) -> Tuple[str, str, str]:
        pass

    def build_turn_from_inputs(self, inputs: TaskInput) -> DatasetTurn:
        pass

    def parse_and_produce_output(self, prompt: str, input_turn: DatasetTurn, completions: Dict[str, float],
                                 inputs: TaskInput, examples: List[DatasetTurn]) -> TaskOutput:
        pass

    def produce_index_turn(self, input_turn: DatasetTurn, inputs: TaskInput, outputs: TaskOutput) -> DatasetTurn:
        pass

    def __init__(self, output_type: OutputType, **kwargs) -> None:
        self.output_type = output_type

    def __call__(self, inputs: List[Any]):
        output: GenericOutputs
        if self.output_type == "dst":
            output: SchemaGuidedDSTOutputs = {
                "schema_belief_state": {'hotel': {'area': 'east'}},
                "active_service_names": ['hotel'],
                "wandb_log_items": {
                    "retrieval_pool_size": 0
                },
                "running_log_items": {}
            }
        elif self.output_type == 'act_tagging':
            output: SchemaGuidedActTaggingOutputs = {
                "system_response_acts": [],
                "active_service_names": [],
                "wandb_log_items": {
                    "retrieval_pool_size": 0
                },
                "running_log_items": {}
            }

        return [output for _ in inputs]


if __name__ == "__main__":
    m = SimpleMockModule("dst")
    print(m(['test']))
