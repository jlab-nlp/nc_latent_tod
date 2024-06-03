import copy
import dataclasses
import logging
import os
import pprint
import sys
import time
from typing import List, Optional, Callable, Tuple, Dict, Any, TypeVar

import wandb
from datasets import Dataset
from nc_latent_tod.mwzeval import Evaluator

from nc_latent_tod.experiments.offline_labelling_experiment import get_prompt_generator_from_cfg
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer

from nc_latent_tod.db.abstract_db import AbstractDB
from tqdm import tqdm

from nc_latent_tod.acts.act import Act
from nc_latent_tod.data_types import DatasetTurn, SchemaBeliefState, DatasetTurnLog
from nc_latent_tod.delex.abstract_delexer import AbstractDelexer
from nc_latent_tod.delex.fuzzy_regex_delexer import FuzzyRegexDelexer
from nc_latent_tod.evaluation.dialogue_act import ActPredictionEvaluator
from nc_latent_tod.evaluation.dst import evaluate_jga
from nc_latent_tod.experiments.abstract_experiment import AbstractExperiment
from nc_latent_tod.experiments.batch_client_lm_module import BatchLMClientDSTModule, AbstractLMClientModule, \
    BatchLMClientPolicyModule, BatchLMClientResponseGenModule
from nc_latent_tod.experiments.config import OnlineE2EEvalLMExperimentConfig
from nc_latent_tod.experiments.data_types import SchemaGuidedDSTOutputs, SchemaGuidedPolicyOutputs, RGEvaluationInput, \
    SchemaGuidedDSTInputs, \
    SchemaGuidedPolicyInputs, SchemaGuidedResponseGenInputs, SchemaGuidedResponseGenOutputs
from nc_latent_tod.experiments.utils import get_turn_ordered_batch_iterator, load_datasets_from_cfg, \
    read_experiment_config, get_db_from_schema_config, \
    ModelCachingModuleBuilder, scale_batch_size_to_memory
from nc_latent_tod.kwargs_prompt.prompt import KwargsPromptGenerator
from nc_latent_tod.acts.utils import get_acts_from_system_acts
from nc_latent_tod.ontology.abstract_ontology import AbstractDBOntology
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.utils.dialogue_states import compute_delta, remove_blank_values
from nc_latent_tod.utils.general import write_json, FunctionsAsNamesEncoder
from nc_latent_tod.utils.state_recorder import PredictionRecorder


class OnlineE2EExperiment(AbstractExperiment):
    """
    Conducts an evaluation experiment of an end-to-end DST system, consisting of:

    1. Predict DST, store prediction, evaluate and log results
    2. Predict Act as system policy, store prediction, evaluate and log results
    3. Predict a de-lexified response, store prediction, evaluate and log results
    """

    dst_prediction_recorder: PredictionRecorder[SchemaBeliefState]
    act_prediction_recorder: PredictionRecorder[List[Act]]
    resp_prediction_recorder: PredictionRecorder[str]
    dst_module: BatchLMClientDSTModule
    policy_module: BatchLMClientPolicyModule
    response_gen_module: BatchLMClientResponseGenModule
    act_evaluator: ActPredictionEvaluator
    dst_correct: int = 0
    dst_turns_correct: int = 0
    n_total: int = 0
    e2e_evaluator: Evaluator

    def __init__(self, *, test_set: Dataset, train_set: Optional[Dataset] = None, output_dir: str = None,
                 schema_loader: Callable[[], List[ServiceSchema]] = None, batch_size: int = 1,
                 dst_module: BatchLMClientDSTModule = None, policy_module: BatchLMClientPolicyModule = None,
                 response_gen_module: BatchLMClientResponseGenModule = None,
                 delexer: AbstractDelexer = None,
                 normalizer: AbstractNormalizer = None,
                 **kwargs) -> None:
        super().__init__(test_set=test_set, train_set=train_set, output_dir=output_dir, schema_loader=schema_loader,
                         batch_size=batch_size, delexer=delexer, normalizer=normalizer, **kwargs)
        self.dst_module = dst_module
        self.policy_module = policy_module
        self.response_gen_module = response_gen_module
        self.dst_prediction_recorder = PredictionRecorder()
        self.act_prediction_recorder = PredictionRecorder()
        self.resp_prediction_recorder = PredictionRecorder()
        self.act_evaluator = ActPredictionEvaluator()
        self.dst_correct = 0
        self.dst_turns_correct = 0
        self.n_total = 0
        self.e2e_evaluator = Evaluator(bleu=True, success=True, richness=True, dst=True)

    def prepare_dst_inputs(self, turn: DatasetTurn) -> SchemaGuidedDSTInputs:
        """
        Prepare inputs for DST module
        """
        dialogue_id: str = turn['dialogue_id']
        turn_id: int = turn['turn_id']
        belief_state_history: List[SchemaBeliefState] = []
        if turn_id > 0:
            belief_state_history = copy.deepcopy(
                self.dst_prediction_recorder.retrieve_previous_turn_prediction(dialogue_id, turn_id))
        if len(belief_state_history) != turn_id:
            raise RuntimeError(f"belief states not being predicted in a proper order. For "
                               f"dialogue={dialogue_id}, turn={turn_id}, expected {turn_id} recorded predictions")
        return {
            "schema": self.schema,
            # history is turns *prior* to this one. Include the user utterance as part of the 'history' here
            "user_utterances": turn['user_utterances'],
            "system_utterances": turn['system_utterances'],
            "belief_state_history": belief_state_history,
        }

    def prepare_act_inputs(self, turn: DatasetTurn, dst_input: SchemaGuidedDSTInputs,
                           dst_output: SchemaGuidedDSTOutputs) -> SchemaGuidedPolicyInputs:
        previous_act_pred: List[Act] = []
        assert turn['user_utterances'] == dst_input['user_utterances']
        if turn['turn_id'] > 0:
            previous_act_pred = self.act_prediction_recorder.retrieve_previous_turn_prediction(turn['dialogue_id'],
                                                                                               turn['turn_id'])[-1]
        inputs: SchemaGuidedPolicyInputs = dict(
            schema=self.schema,
            last_system_acts=previous_act_pred,
            user_utterances=turn['user_utterances'],
            system_utterances=turn['system_utterances'],
            prior_state=dst_input['belief_state_history'][-1] if dst_input['belief_state_history'] else {},
            next_state=dst_output['schema_belief_state'],
        )
        return inputs

    @staticmethod
    def prepare_response_gen_inputs(turn: DatasetTurn, act_input: SchemaGuidedPolicyInputs,
                                    act_output: SchemaGuidedPolicyOutputs) -> SchemaGuidedResponseGenInputs:
        # this will be valid so long as the only difference in available inputs between RG and policy is the policy's
        # prediction
        # noinspection PyTypeChecker
        return {
            **act_input,
            "system_response_acts": act_output['system_response_acts'],
        }

    def print_log_and_store_dst_result(self, log: DatasetTurnLog, dst_input: SchemaGuidedDSTInputs,
                                       dst_output: SchemaGuidedDSTOutputs) -> DatasetTurnLog:
        predicted_prior_context: SchemaBeliefState = dst_input['belief_state_history'][-1] if dst_input[
            'belief_state_history'] else {}
        predicted_slot_values: SchemaBeliefState = dst_output['schema_belief_state']

        # add DST items to log
        log['pred_belief_state'] = predicted_slot_values
        predicted_delta = compute_delta(predicted_prior_context, predicted_slot_values)
        log['pred_delta_slot_values'] = predicted_delta
        log['pred_prior_context'] = predicted_prior_context

        # evaluate if possible:
        gold_delta: Optional[SchemaBeliefState] = None
        gold_state: Optional[SchemaBeliefState] = None
        if 'slot_values' in log:
            gold_state = remove_blank_values(log['slot_values'])
            gold_delta = remove_blank_values(log['turn_slot_values'])
            this_jga: float = evaluate_jga(prediction=predicted_slot_values, gold=gold_state)
            self.dst_correct += int(this_jga)
            turn_goal_accuracy: float = evaluate_jga(prediction=predicted_delta,
                                                     gold=gold_delta)
            self.dst_turns_correct += int(turn_goal_accuracy)

            # save log turn-level evaluation results
            log['jga'] = this_jga
            log['turn_acc'] = turn_goal_accuracy

            # log evaluation results
            self.logger.log({
                "current_jga": self.dst_correct / self.n_total,
                "current_turn_acc": self.dst_turns_correct / self.n_total,
            })

            # print evaluation results
            is_correct_label_str: str = "correct!" if this_jga else "wrong!"
            print(f"\n===================== DST: {is_correct_label_str} =======================")
            print(f"this turn acc: {turn_goal_accuracy}")
            print(f"avg turn acc: {self.dst_turns_correct / self.n_total}, avg jga: {self.dst_correct / self.n_total}")

        # print results
        print(f"user: {log['user_utterances'][-1]}")
        print(f"DST prediction: {pprint.pformat(predicted_slot_values)}")
        completions = dst_output['running_log_items'].get('all_completions')
        print(f"DST completions: {pprint.pformat(completions)}")
        if gold_delta is not None:
            print(f"DST gold turn change: {pprint.pformat(gold_delta)}")
            print(f"DST delta difference: {pprint.pformat(compute_delta(predicted_delta, gold_delta))}")
        if gold_state is not None:
            print(f"DST total difference: {pprint.pformat(compute_delta(predicted_slot_values, gold_state))}")

        # store our prediction for DST:
        self.dst_prediction_recorder.add_prediction(
            dialogue_id=log['dialogue_id'], turn_id=log['turn_id'], prediction=predicted_slot_values
        )

        # return the log
        return log

    def print_log_and_store_act_result(self, log: DatasetTurnLog, act_input: SchemaGuidedPolicyInputs,
                                       act_output: SchemaGuidedPolicyOutputs) -> DatasetTurnLog:

        # add Act items to log (leave in serialized state)
        log['pred_system_response_acts'] = act_output['system_response_acts']
        log['pred_act_based_active_service_names'] = act_output['active_service_names']

        # de-serialize for evaluation and printing
        predicted_acts: List[Act] = get_acts_from_system_acts(act_output['system_response_acts'], self.schema)
        # evaluate if possible:
        print(f"\n===================== Policy =======================")
        print(f"predicted turn acts: {pprint.pformat(predicted_acts)}")
        if 'system_response_acts' in log:
            gold_acts: List[Act] = get_acts_from_system_acts(log['system_response_acts'], self.schema)
            print(f"gold turn acts: {pprint.pformat(gold_acts)}")
            these_scores: Dict[str, float] = self.act_evaluator.evaluate_turn(
                pred_turn_acts=predicted_acts,
                gold_turn_acts=gold_acts
            )
            print(f"This turn scores: {these_scores}")
            self.act_evaluator.add_turn(
                pred_turn_acts=predicted_acts,
                gold_turn_acts=gold_acts
            )
            self.logger.log(self.act_evaluator.current_scores())

        # store prediction for later use as inputs
        self.act_prediction_recorder.add_prediction(
            dialogue_id=log['dialogue_id'], turn_id=log['turn_id'], prediction=predicted_acts
        )

        # return log
        return log

    def print_log_and_store_delex_result(self, log: DatasetTurnLog, delex_system_response: str) -> DatasetTurnLog:
        # write to log:
        log['pred_delex_system_response'] = delex_system_response

        print(f"true system response: {log.get('system_response', '<unknown>')}")
        print(f"delex pred system response: {delex_system_response}")

        # not sure we actually use these in offline labelling
        self.resp_prediction_recorder.add_prediction(
            dialogue_id=log['dialogue_id'], turn_id=log['turn_id'], prediction=delex_system_response
        )

        return log

    def run(self) -> Tuple[List[DatasetTurnLog], Dict[str, Any]]:
        """
        Run the experiment, returning the running log and final stats

        This includes:
            1. Predict DST, evaluate if possible, store prediction and log results
            2. Predict Act, evaluate if possible, store prediction and log results
            3. Delex response using act, evaluate if possible, store prediction and log results
        """
        selected_set: Dataset = self.test_set

        # start experiment
        run_start: float = time.time()

        for turns in tqdm(get_turn_ordered_batch_iterator(selected_set, self.batch_size),
                          desc="Evaluating in order of increasing turn_id"):
            batch_start: float = time.time()
            batch_size: int = len(turns)
            turns: List[DatasetTurn]

            # the version we write to
            logs: List[DatasetTurnLog] = copy.deepcopy(turns)

            # 1. Run the DST Module
            batch_dst_inputs: List[SchemaGuidedDSTInputs] = [self.prepare_dst_inputs(turn) for turn in turns]
            batch_dst_outputs: List[SchemaGuidedDSTOutputs] = self.dst_module(batch_dst_inputs)

            # 2. Run the Act Policy Module
            batch_act_inputs: List[SchemaGuidedPolicyInputs] = [
                self.prepare_act_inputs(turn, dst_input, dst_output)
                for turn, dst_input, dst_output in zip(turns, batch_dst_inputs, batch_dst_outputs)
            ]
            batch_act_outputs: List[SchemaGuidedPolicyOutputs] = self.policy_module(batch_act_inputs)

            # 3. Run the Response Generation Module
            batch_resp_inputs: List[SchemaGuidedResponseGenInputs] = [
                self.prepare_response_gen_inputs(turn, act_input, act_output)
                for turn, act_input, act_output in zip(turns, batch_act_inputs, batch_act_outputs)
            ]
            batch_resp_outputs: List[SchemaGuidedResponseGenOutputs] = self.response_gen_module(batch_resp_inputs)

            # process outputs from both sequentially (Delex within loop)
            for turn, dst_input, dst_output, act_input, act_output, resp_input, resp_output, log in zip(
                    turns,
                    batch_dst_inputs,
                    batch_dst_outputs,
                    batch_act_inputs,
                    batch_act_outputs,
                    batch_resp_inputs,
                    batch_resp_outputs,
                    logs, strict=True):
                self.n_total += 1
                dialogue_id: str = log['dialogue_id']
                turn_id: int = log['turn_id']

                # assume no collisions, revisit if needed
                for outputs in (dst_output, act_output):
                    if 'wandb_log_items' in outputs:
                        self.logger.log(outputs['wandb_log_items'])
                    if 'running_log_items' in outputs:
                        log.update(outputs['running_log_items'])

                # record the DST and ACT predictions
                print(f"this is the {self.n_total - 1}th example. {dialogue_id}_turn_{turn_id}")

                log = self.print_log_and_store_dst_result(log, dst_input, dst_output)
                log = self.print_log_and_store_act_result(log, act_input, act_output)

                # delexify (again, in case we predict a lexicalized response from the policy)
                pred_system_response: str = resp_output['system_response']
                predicted_acts: List[Act] = get_acts_from_system_acts(act_output['system_response_acts'], self.schema)
                delex_system_response: str = self.delexer.delexify(pred_system_response, system_acts=predicted_acts)
                log = self.print_log_and_store_delex_result(log, delex_system_response)

                if 'pred_delex_system_response' not in log:
                    raise ValueError(f"expected a delex system response, but got: {log['pred_delex_system_response']}")

                # add the log to the running log
                self.turn_logger.log_turn(
                    log,
                    dst_completion=dst_output['running_log_items'].get('completion'),
                    dst_gold_delta=turn['turn_slot_values'],
                    jga=log['jga'],
                    turn_acc=log['turn_acc'],
                    policy_completion=act_output['running_log_items'].get('completion'),
                    belief_state=log['pred_belief_state'],
                    policy_gold=turn['system_response_acts'],
                    schema=self.schema,
                )

                # log other information and step
                self.logger.log({
                    "n_total": self.n_total,
                    "batch_samples": batch_size,
                    "batch_samples_per_second": batch_size / (time.time() - batch_start),
                    "avg_samples_per_second": self.n_total / (time.time() - run_start)
                })
                self.logger.step()
                print("\n")

                # write out running log regularly, in-case we stop a run. Give some buffer in-case we
                # accidentally start, and didn't want to over-write
                if self.n_total > 20:
                    self.turn_logger.write_running_log(path=os.path.join(self.output_dir, "running_log.json"))

        # prepare inputs for mwzeval
        eval_input: RGEvaluationInput = self.build_rg_evaluation_input(self.turn_logger.running_log)

        # compute e2e scores
        scores = self.e2e_evaluator.evaluate(eval_input)
        print(f"e2e scores: {scores}")

        stats = {
            # DST stats
            'jga': self.dst_correct / self.n_total,
            "turn_acc": self.dst_turns_correct / self.n_total,
            # Act tagging stats
            **{f"final_{k}": v for k, v in self.act_evaluator.current_scores().items()},
            # E2E stats
            **scores
        }

        self.logger.log(stats)
        self.logger.step()
        return self.turn_logger.running_log, stats


def validate_config(run_config: OnlineE2EEvalLMExperimentConfig) -> None:
    if run_config.create_self_labelled_dataset:
        assert run_config.publish_labelled_dataset_as and run_config.publish_labelled_dataset_as.path_or_name, \
            "must specify a location to publish to in run_config.publish_labelled_dataset_as"


T = TypeVar('T', bound=AbstractLMClientModule)


def main(run_config: OnlineE2EEvalLMExperimentConfig, **kwargs) -> None:
    validate_config(run_config)

    for module in (run_config.dst, run_config.policy, run_config.response_gen):
        module.model.batch_size = scale_batch_size_to_memory(module.model.batch_size)
    # create the output folder
    os.makedirs(run_config.output.output_dir, exist_ok=True)

    # write out this experiment's configuration
    write_json(dataclasses.asdict(run_config),
               os.path.join(run_config.output.output_dir, "exp_config.json"),
               # serialize functions by their names, we don't need to de-serialize them
               indent=4, cls=FunctionsAsNamesEncoder)

    # read data and verify expectations
    train_set, eval_set = load_datasets_from_cfg(run_config.data)
    assert not train_set, "this is a labelling experiment, no training!"

    # config includes lazy loading provider functions
    schema: List[ServiceSchema] = run_config.schema_config.schema_loader()
    ontology: AbstractDBOntology = run_config.schema_config.ontology()
    db: AbstractDB = get_db_from_schema_config(run_config.schema_config)

    # build our prompt generator
    prompt_generator: KwargsPromptGenerator = get_prompt_generator_from_cfg(
        run_config=run_config,
        schema=schema,
        ontology=ontology,
        db=db
    )
    builder = ModelCachingModuleBuilder()
    dst_model: BatchLMClientDSTModule = builder.build_batch_module_from_config(run_config.dst, prompt_generator)
    policy_model: BatchLMClientPolicyModule = builder.build_batch_module_from_config(run_config.policy, prompt_generator)
    rg_model: BatchLMClientResponseGenModule = builder.build_batch_module_from_config(run_config.response_gen, prompt_generator)

    largest_model_batch_size: int = min([
        scale_batch_size_to_memory(run_config.dst.model.batch_size),
        scale_batch_size_to_memory(run_config.policy.model.batch_size),
        scale_batch_size_to_memory(run_config.response_gen.model.batch_size)
    ])
    largest_best_of: int = run_config.dst.generation_cfg.sampling_args.get('best_of', 1) if \
                               run_config.dst.generation_cfg.generation_mode != 'greedy' else 1

    logging.info(f"largest model batch size: {largest_model_batch_size}, largest best_of: {largest_best_of}")
    overall_batch_size: int = largest_model_batch_size // min(largest_best_of, largest_model_batch_size)
    experiment: OnlineE2EExperiment = OnlineE2EExperiment(
        test_set=eval_set,
        train_set=None,
        output_dir=run_config.output.output_dir,
        schema_loader=run_config.schema_config.schema_loader,
        batch_size=overall_batch_size,
        dst_module=dst_model,
        policy_module=policy_model,
        response_gen_module=rg_model,
        delexer=FuzzyRegexDelexer(),
        **kwargs
    )
    running_log: List[DatasetTurnLog] = []
    stats = None
    try:
        running_log, stats = experiment.run()
        if run_config.create_self_labelled_dataset:
            raise NotImplementedError("do I need this?")
    finally:
        if running_log and len(running_log) == len(eval_set):
            run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            run.tags.append("complete_run")
            run.update()
        artifact: wandb.Artifact = wandb.Artifact(wandb.run.name, type="run_output")
        artifact.add_file(os.path.join(run_config.output.output_dir, "running_log.json"))
        artifact.add_file(os.path.join(run_config.output.output_dir, "exp_config.json"))
        wandb.log_artifact(artifact)


if __name__ == '__main__':
    cfg: OnlineE2EEvalLMExperimentConfig = read_experiment_config(sys.argv[1],
                                                                  data_class=OnlineE2EEvalLMExperimentConfig)
    wandb.init(
        config=dataclasses.asdict(cfg),
        project="nc_latent_tod", entity=os.environ.get("WANDB_ENTITY", "kingb12"),
        name=cfg.wandb.run_name,
        notes=cfg.wandb.run_notes,
        group=cfg.wandb.run_group,
        tags=cfg.wandb.run_tags
    )
    main(cfg)
