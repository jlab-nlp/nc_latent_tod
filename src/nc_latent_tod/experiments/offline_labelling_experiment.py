import copy
import dataclasses
import json
import logging
import os
import pprint
import sys
import tempfile
import time
from typing import List, Optional, Callable, Tuple, Dict, Any, TypeVar

import wandb
from datasets import Dataset
from tqdm import tqdm
from wandb.sdk.lib.runid import generate_id

from nc_latent_tod.acts.act import Act
from nc_latent_tod.acts.utils import get_acts_from_system_acts
from nc_latent_tod.data_types import DatasetTurn, SchemaBeliefState, DatasetTurnLog
from nc_latent_tod.db.abstract_db import AbstractDB
from nc_latent_tod.delex.abstract_delexer import AbstractDelexer
from nc_latent_tod.delex.fuzzy_regex_delexer import FuzzyRegexDelexer
from nc_latent_tod.evaluation.dialogue_act import ActPredictionEvaluator
from nc_latent_tod.evaluation.dst import evaluate_jga
from nc_latent_tod.experiments.abstract_experiment import AbstractExperiment
from nc_latent_tod.experiments.batch_client_lm_module import BatchLMClientDSTModule, BatchLMClientActTagModule, \
    AbstractLMClientModule
from nc_latent_tod.experiments.config import OfflineLabellingLMExperimentConfig
from nc_latent_tod.experiments.data_types import OfflineSchemaGuidedDSTInputs, SchemaGuidedDSTOutputs, \
    SchemaGuidedActTaggingInputs, SchemaGuidedActTaggingOutputs, RGEvaluationInput
from nc_latent_tod.experiments.dataset_uploads import apply_self_labels, \
    build_and_upload_self_labeled_dataset
from nc_latent_tod.experiments.dynamo_manifest import DynamoDBExperimentLogsManifest
from nc_latent_tod.experiments.manifest import ExperimentLogsManifest
from nc_latent_tod.experiments.mock_simple_module import SimpleMockModule
from nc_latent_tod.experiments.utils import get_turn_ordered_batch_iterator, load_datasets_from_cfg, \
    read_experiment_config, get_db_from_schema_config, \
    verify_cuda_is_available_if_needed, ModelCachingModuleBuilder, scale_batch_size_to_memory
from nc_latent_tod.kwargs_prompt.prompt import KwargsPromptGenerator
from nc_latent_tod.kwargs_prompt.simple_ft_prompt import SimpleFTKwargsPromptGenerator
from nc_latent_tod.mwzeval import Evaluator
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.ontology.abstract_ontology import AbstractDBOntology
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.utils import artifacts
from nc_latent_tod.utils.dialogue_states import compute_delta, remove_blank_values
from nc_latent_tod.utils.general import write_json, FunctionsAsNamesEncoder
from nc_latent_tod.utils.state_recorder import PredictionRecorder


class OfflineLabellingExperiment(AbstractExperiment):
    """
    Conducts an offline labelling experiment using the given modules as the DST system, Act Tagger, and Delexer. This
    consists of:

    1. Label DST on training data, reporting JGA/turn_acc for debug if computable (i.e. if the dataset is pre-labelled)
    2. Label Acts on training data, reporting Act/Slot/Value F1s if computable(i.e. if the dataset is pre-labelled)
    3. Label Delex responses, computing end-to-end scores if computable (i.e. if the dataset is pre-labelled)
    4. Upload the pseudo-labelled training data to huggingface, if specified.
    """

    dst_prediction_recorder: PredictionRecorder[SchemaBeliefState]
    act_prediction_recorder: PredictionRecorder[List[Act]]
    resp_prediction_recorder: PredictionRecorder[str]
    dst_module: BatchLMClientDSTModule
    act_module: BatchLMClientActTagModule
    act_evaluator: ActPredictionEvaluator
    batch_size: int
    data_warmup: int
    data_num_partitions: int
    dst_correct: int = 0
    dst_turns_correct: int = 0
    n_total: int = 0
    e2e_evaluator: Evaluator
    resume_from_logs: Optional[List[DatasetTurnLog]]

    def __init__(self, *, test_set: Dataset, train_set: Optional[Dataset] = None, output_dir: str = None,
                 schema_loader: Callable[[], List[ServiceSchema]] = None, batch_size: int = 1,
                 dst_module: BatchLMClientDSTModule = None, act_module: BatchLMClientActTagModule = None,
                 delexer: AbstractDelexer = None,
                 normalizer: AbstractNormalizer = None,
                 data_warmup: int = 0,
                 data_num_partitions: int = 0,
                 resume_from_logs: Optional[List[DatasetTurnLog]] = None,
                 **kwargs) -> None:
        super().__init__(test_set=test_set, train_set=train_set, output_dir=output_dir, schema_loader=schema_loader,
                         batch_size=batch_size, delexer=delexer, normalizer=normalizer, **kwargs)
        self.data_warmup = data_warmup
        self.data_num_partitions = data_num_partitions
        self.dst_module = dst_module
        self.act_module = act_module
        self.dst_prediction_recorder = PredictionRecorder()
        self.act_prediction_recorder = PredictionRecorder()
        self.resp_prediction_recorder = PredictionRecorder()
        self.act_evaluator = ActPredictionEvaluator()
        self.dst_correct = 0
        self.dst_turns_correct = 0
        self.n_total = 0
        self.e2e_evaluator = Evaluator(bleu=True, success=True, richness=True, dst=True)
        self.resume_from_logs = resume_from_logs

    def prepare_dst_inputs(self, turn: DatasetTurn) -> OfflineSchemaGuidedDSTInputs:
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
            "system_response": turn['system_response'],
            "system_response_acts": turn['system_response_acts'],
            "last_system_response_acts": None,
        }

    def prepare_act_inputs(self, turn: DatasetTurn, dst_input: OfflineSchemaGuidedDSTInputs,
                           dst_output: SchemaGuidedDSTOutputs) -> SchemaGuidedActTaggingInputs:
        previous_act_pred: List[Act] = []
        assert turn['user_utterances'] == dst_input['user_utterances']
        if turn['turn_id'] > 0:
            previous_act_pred = self.act_prediction_recorder.retrieve_previous_turn_prediction(turn['dialogue_id'],
                                                                                               turn['turn_id'])[-1]
        inputs: SchemaGuidedActTaggingInputs = dict(
            schema=self.schema,
            last_system_acts=previous_act_pred,
            system_response=turn['system_response'],
            user_utterances=turn['user_utterances'],
            system_utterances=turn['system_utterances'],
            prior_state=dst_input['belief_state_history'][-1] if dst_input['belief_state_history'] else {},
            next_state=dst_output['schema_belief_state'],
        )
        return inputs

    def replay_logs(self, resume_from_logs: List[DatasetTurnLog]):
        since_written: int = 0
        for turn in tqdm(resume_from_logs, desc="Replaying resumed logs", total=len(resume_from_logs)):

            # the version we write to
            log: DatasetTurnLog = copy.deepcopy(turn)

            self.n_total += 1
            since_written += 1

            dialogue_id: str = log['dialogue_id']
            turn_id: int = log['turn_id']

            # assume no collisions, revisit if needed
            for outputs in (dst_output, act_output):
                if 'wandb_log_items' in outputs:
                    self.logger.log(outputs['wandb_log_items'])
                if 'running_log_items' in outputs:
                    log.update(outputs['running_log_items'])

            # record the DST and ACT predictions
            print(f"(replay) this is the {self.n_total - 1}th example. {dialogue_id}_turn_{turn_id}")

            dst_input: OfflineSchemaGuidedDSTInputs = self.prepare_dst_inputs(turn)
            pred_state: SchemaBeliefState = remove_blank_values(log['pred_belief_state'])
            dst_output: SchemaGuidedDSTOutputs = {
                "schema_belief_state": pred_state,
                "active_service_names": list(pred_state.keys()),
                "running_log_items": {},
                "wandb_log_items": {}
            }
            log = self.print_log_and_store_dst_result(log, dst_input, dst_output)
            act_input: SchemaGuidedActTaggingInputs = self.prepare_act_inputs(turn, dst_input, dst_output)
            act_output: SchemaGuidedActTaggingOutputs = {
                "system_response_acts": log['pred_system_response_acts'] or [],
                "active_service_names": log['pred_act_based_active_service_names'],
                "running_log_items": {},
                "wandb_log_items": {}
            }
            log = self.print_log_and_store_act_result(log, act_input, act_output)

            # delexify
            predicted_acts: List[Act] = get_acts_from_system_acts(act_output['system_response_acts'], self.schema)
            try:
                delex_system_response: str = self.delexer.delexify(turn['system_response'],
                                                                   system_acts=predicted_acts)
            except BaseException as e:
                logging.error("Unable to delexify", e)
                log['not_valid'] = 1
                delex_system_response = turn['system_response']
            log = self.print_log_and_store_delex_result(log, delex_system_response)

            # add the log to the running log
            gold_acts: List[Act] = get_acts_from_system_acts(turn['system_response_acts'], self.schema)
            self.turn_logger.log_turn(
                log,
                # see kwargs in DSTTextFileLogViewer
                dst_completion=dst_output['running_log_items'].get('completion'),
                dst_gold_delta=turn['turn_slot_values'],
                jga=log['jga'],
                turn_acc=log['turn_acc'],
                # see kwargs in ActTaggingTextFileLogViewer
                schema=self.schema,
                act_tag_completion=act_output['running_log_items'].get('completion'),
                act_tags_gold=get_acts_from_system_acts(turn['system_response_acts'], self.schema),
                belief_state=log['pred_belief_state'],
                gold_delex_system_response=self.delexer.delexify(turn['system_response'], system_acts=gold_acts),
            )

            # log other information and step
            self.logger.log({
                "n_total": self.n_total,
                "batch_samples": 1,
                "batch_samples_per_second": -1,
                "avg_samples_per_second": -1
            })
            self.logger.step()
            print("\n")

            # write out running log regularly, in-case we stop a run. Give some buffer in-case we
            # accidentally start, and didn't want to over-write
            if since_written > 100:
                self.turn_logger.write_running_log(path=os.path.join(self.output_dir, "running_log.json"))
                since_written = 0
            if self.dst_module.retriever and self.dst_module.retriever_add_to_index:
                self.dst_module.retriever.add_items([turn])
            if self.act_module.retriever and self.act_module.retriever_add_to_index:
                self.act_module.retriever.add_items([turn])

    def print_log_and_store_dst_result(self, log: DatasetTurnLog, dst_input: OfflineSchemaGuidedDSTInputs,
                                       dst_output: SchemaGuidedDSTOutputs) -> DatasetTurnLog:
        predicted_prior_context: SchemaBeliefState = dst_input['belief_state_history'][-1] if \
            dst_input['belief_state_history'] else {}
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
            # write eval values to log
            log['jga'] = this_jga
            log['turn_acc'] = turn_goal_accuracy

            self.dst_turns_correct += int(turn_goal_accuracy)

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

    def print_log_and_store_act_result(self, log: DatasetTurnLog, act_input: SchemaGuidedActTaggingInputs,
                                       act_output: SchemaGuidedActTaggingOutputs) -> DatasetTurnLog:

        # add Act items to log (leave in serialized state)
        log['pred_system_response_acts'] = act_output['system_response_acts']
        log['pred_act_based_active_service_names'] = act_output['active_service_names']

        # de-serialize for evaluation and printing
        predicted_acts: List[Act] = get_acts_from_system_acts(act_output['system_response_acts'], self.schema)
        # evaluate if possible:
        print(f"\n===================== Act Tagging =======================")
        print(f"act completions: {pprint.pformat(act_output['running_log_items'].get('all_completions'))}")
        print(f"predicted turn acts: {pprint.pformat(predicted_acts)}")
        if 'system_response_acts' in log:
            gold_acts: List[Act] = get_acts_from_system_acts(log['system_response_acts'], self.schema)
            print(f"gold turn acts: {pprint.pformat(gold_acts)}")
            try:
                these_scores: Dict[str, float] = self.act_evaluator.evaluate_turn(
                    pred_turn_acts=predicted_acts,
                    gold_turn_acts=gold_acts
                )
                print(f"This turn scores: {these_scores}")
                self.act_evaluator.add_turn(
                    pred_turn_acts=predicted_acts,
                    gold_turn_acts=gold_acts
                )
            except BaseException as e:
                logging.error(f"error evaluating turn acts", e)
                print(f"This turn scores: unable to score!: {repr(predicted_acts)}")
                self.act_evaluator.add_turn(
                    pred_turn_acts=[],  # since we call the same function for evaluation, it would just fail again
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

        print(f"true system response: {log['system_response']}")
        print(f"delex system response: {delex_system_response}")

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
        if self.resume_from_logs:
            self.replay_logs(self.resume_from_logs)
        selected_set: Dataset = self.test_set

        # start experiment
        run_start: float = time.time()
        since_written: int = 0
        data_iterator = get_turn_ordered_batch_iterator(
            selected_set,
            batch_size=self.batch_size,
            num_partitions=self.data_num_partitions,
            warmup=self.data_warmup,
            skip_turns_in=self.resume_from_logs
        )
        for turns in tqdm(data_iterator, desc="Evaluating in order of increasing turn_id"):
            batch_start: float = time.time()
            batch_size: int = len(turns)
            turns: List[DatasetTurn]

            # the version we write to
            logs: List[DatasetTurnLog] = copy.deepcopy(turns)

            # 1. Run the DST Module
            batch_dst_inputs: List[OfflineSchemaGuidedDSTInputs] = [self.prepare_dst_inputs(turn) for turn in turns]
            batch_dst_outputs: List[SchemaGuidedDSTOutputs] = self.dst_module(batch_dst_inputs)

            # 2. Run the Act Tagging Module
            batch_act_inputs: List[SchemaGuidedActTaggingInputs] = [
                self.prepare_act_inputs(turn, dst_input, dst_output)
                for turn, dst_input, dst_output in zip(turns, batch_dst_inputs, batch_dst_outputs)
            ]
            batch_act_outputs: List[SchemaGuidedActTaggingOutputs] = self.act_module(batch_act_inputs)

            # process outputs from both sequentially (Delex within loop)
            for turn, dst_input, dst_output, act_input, act_output, log in \
                    zip(turns, batch_dst_inputs, batch_dst_outputs, batch_act_inputs, batch_act_outputs, logs,
                        strict=True):
                self.n_total += 1
                since_written += 1

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

                # delexify
                predicted_acts: List[Act] = get_acts_from_system_acts(act_output['system_response_acts'], self.schema)
                try:
                    delex_system_response: str = self.delexer.delexify(turn['system_response'],
                                                                       system_acts=predicted_acts)
                except BaseException as e:
                    logging.error("Unable to delexify", e)
                    log['not_valid'] = 1
                    delex_system_response = turn['system_response']
                log = self.print_log_and_store_delex_result(log, delex_system_response)

                # add the log to the running log
                gold_acts: List[Act] = get_acts_from_system_acts(turn['system_response_acts'], self.schema)
                self.turn_logger.log_turn(
                    log,
                    # see kwargs in DSTTextFileLogViewer
                    dst_completion=dst_output['running_log_items'].get('completion'),
                    dst_gold_delta=turn['turn_slot_values'],
                    jga=log['jga'],
                    turn_acc=log['turn_acc'],
                    # see kwargs in ActTaggingTextFileLogViewer
                    schema=self.schema,
                    act_tag_completion=act_output['running_log_items'].get('completion'),
                    act_tags_gold=get_acts_from_system_acts(turn['system_response_acts'], self.schema),
                    belief_state=log['pred_belief_state'],
                    gold_delex_system_response=self.delexer.delexify(turn['system_response'], system_acts=gold_acts),
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
                if since_written > 100:
                    self.turn_logger.write_running_log(path=os.path.join(self.output_dir, "running_log.json"))
                    since_written = 0

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

        self.turn_logger.write_running_log(path=os.path.join(self.output_dir, "running_log.json"))

        self.logger.log(stats)
        self.logger.step()
        return self.turn_logger.running_log, stats


def validate_config(run_config: OfflineLabellingLMExperimentConfig) -> None:
    if run_config.create_self_labelled_dataset:
        assert run_config.publish_labelled_dataset_as and run_config.publish_labelled_dataset_as.path_or_name, \
            "must specify a location to publish to in run_config.publish_labelled_dataset_as"


T = TypeVar('T', bound=AbstractLMClientModule)


def get_prompt_generator_from_cfg(run_config: OfflineLabellingLMExperimentConfig, schema: List[ServiceSchema],
                                  ontology: AbstractDBOntology, db: AbstractDB) -> KwargsPromptGenerator:
    if not hasattr(run_config, "prompt_generator") or not run_config.prompt_generator or \
            run_config.prompt_generator == "kwargs":
        return KwargsPromptGenerator(
            schema=schema,
            ontology=ontology,
            db=db
        )
    elif run_config.prompt_generator == "simple_kwargs":
        return SimpleFTKwargsPromptGenerator(
            schema=schema,
            ontology=ontology,
            db=db
        )
    else:
        raise ValueError(f"invalid prompt generator type: {run_config.prompt_generator}")


def build_offline_label_experiment(run_config: OfflineLabellingLMExperimentConfig,
                                   builder: Optional[ModelCachingModuleBuilder] = None,
                                   manifest: Optional[ExperimentLogsManifest] = None,
                                   **kwargs) -> OfflineLabellingExperiment:
    validate_config(run_config)
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
    if os.environ.get('ALWAYS_LOAD_IN_8BIT', False):
        run_config.dst.model.load_in_8bit = True
        run_config.act_tag.model.load_in_8bit = True

    builder = builder or ModelCachingModuleBuilder()
    if run_config.dst.model.model_type == "mock_simple":
        dst_model = SimpleMockModule("dst")
    else:
        dst_model: BatchLMClientDSTModule = builder.build_batch_module_from_config(
            cfg=run_config.dst,
            prompt_generator=prompt_generator,
            normalizer=(prompt_generator.normalizer if run_config.dst.retriever.use_normalizer else None)
        )
    if run_config.act_tag.model.model_type == "mock_simple":
        act_model = SimpleMockModule("act_tagging")
    else:
        act_model: BatchLMClientActTagModule = builder.build_batch_module_from_config(
            cfg=run_config.act_tag,
            prompt_generator=prompt_generator,
        )

    resume_from_logs: Optional[List[DatasetTurnLog]] = None
    if run_config.resume_from_logs:
        resume_from_logs = artifacts.read_run_artifact_logs(run_config.resume_from_logs)

    if run_config.manifest.seed_retrievers_from_manifest:
        assert manifest, "must specify a manifest to seed from"
        logs: List[DatasetTurnLog] = manifest.get_logs_from_group(run_config.manifest.group_id)
        logging.info(f"seeding retrievers from manifest with {len(logs)} logs")
        dataset_turns: List[DatasetTurn] = apply_self_labels(logs, schema)
        if run_config.dst.retriever.use_retriever:
            dst_model.retriever.add_items(dataset_turns)
        if run_config.act_tag.retriever.use_retriever:
            act_model.retriever.add_items(dataset_turns)

    largest_model_batch_size: int = min(
        scale_batch_size_to_memory(run_config.dst.model.batch_size),
        scale_batch_size_to_memory(run_config.act_tag.model.batch_size)
    )
    largest_best_of: int = max(
        run_config.dst.generation_cfg.sampling_args.get('best_of', 1) if run_config.dst.generation_cfg.generation_mode != 'greedy' else 1,
        run_config.act_tag.generation_cfg.sampling_args.get('best_of', 1) if run_config.act_tag.generation_cfg.generation_mode != 'greedy' else 1
    )
    logging.info(f"largest model batch size: {largest_model_batch_size}, largest best_of: {largest_best_of}")
    overall_batch_size: int = largest_model_batch_size // min(largest_best_of, largest_model_batch_size)
    logging.info(f"overall batch size: {overall_batch_size}")
    experiment: OfflineLabellingExperiment = OfflineLabellingExperiment(
        test_set=eval_set,
        train_set=None,
        output_dir=run_config.output.output_dir,
        schema_loader=run_config.schema_config.schema_loader,
        # This is how many examples we choose to label at a time, but one module may be able to parallelize better
        batch_size=overall_batch_size,
        data_warmup=run_config.data_warmup,
        data_num_partitions=run_config.data_num_partitions,
        dst_module=dst_model,
        act_module=act_model,
        resume_from_logs=resume_from_logs,
        delexer=FuzzyRegexDelexer(),
        **kwargs
    )
    return experiment


def main(run_config: OfflineLabellingLMExperimentConfig,
         builder: Optional[ModelCachingModuleBuilder] = None,
         **kwargs) -> None:
    os.makedirs(run_config.output.output_dir, exist_ok=True)
    manifest: Optional[ExperimentLogsManifest] = None
    for module in (run_config.dst, run_config.act_tag):
        module.model.batch_size = scale_batch_size_to_memory(module.model.batch_size)
    if run_config.manifest.write_to_manifest or run_config.manifest.seed_retrievers_from_manifest:
        assert run_config.manifest.manifest_path, "must specify a manifest path to write to"
        if run_config.manifest.type == 'local':
            manifest = ExperimentLogsManifest(
                manifest_path=run_config.manifest.manifest_path,
                create_if_not_exists=not run_config.manifest.manifest_must_exist,
                logs_cache_dir=os.environ.get("MANIFEST_LOGS_CACHE_DIR", None)
            )
        elif run_config.manifest.type == 'dynamo':
            manifest = DynamoDBExperimentLogsManifest(
                aws_region='us-west-2',
                table_name=run_config.manifest.manifest_path,
                create_if_not_exists=False,
                logs_cache_dir=os.environ.get("MANIFEST_LOGS_CACHE_DIR", None)
            )
        else:
            raise ValueError(f"invalid manifest type: {run_config.manifest.type}")
        if run_config.manifest.group_id_must_exist:
            group_entries = manifest.read_group(run_config.manifest.group_id)
            assert group_entries is not None, f"group_id={run_config.manifest.group_id} must exist in manifest: group_entries={group_entries}"

    experiment: OfflineLabellingExperiment = build_offline_label_experiment(run_config,
                                                                            builder=builder,
                                                                            manifest=manifest,
                                                                            **kwargs)
    running_log: List[DatasetTurnLog] = []
    stats = None
    try:
        running_log, stats = experiment.run()
        if run_config.create_self_labelled_dataset:
            build_and_upload_self_labeled_dataset(run_config, running_log, experiment.schema)
    finally:
        if running_log and len(running_log) == len(experiment.test_set):
            run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            run.tags.append("complete_run")
            run.update()
            if run_config.manifest.write_to_manifest:
                if not manifest:
                    logging.error("No manifest to write to!")
                else:
                    manifest.add_entry({
                        "run_id": wandb.run.id,
                        "group_id": run_config.manifest.group_id,
                        "num_logs": len(running_log),
                        "labelled_dataset_path_or_name": run_config.data.eval_set_path_or_name,
                        "labelled_dataset_split_name": run_config.data.eval_set_split_name
                    })
                    manifest_path: str = run_config.manifest.manifest_path
                    if run_config.manifest.type != 'local':
                        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                            manifest_path = temp_file.name
                            data = {run_config.manifest.group_id: manifest.read_group(run_config.manifest.group_id)}
                            json.dump(data, temp_file, indent=4)
                    # add as an artifact, in case everything breaks
                    artifact: wandb.Artifact = wandb.Artifact(f"manifest_{wandb.run.id}", type="manifest")
                    artifact.add_file(manifest_path)
                    wandb.log_artifact(artifact)

        artifact: wandb.Artifact = wandb.Artifact(wandb.run.id, type="run_output")
        artifact.add_file(os.path.join(run_config.output.output_dir, "running_log.json"))
        artifact.add_file(os.path.join(run_config.output.output_dir, "exp_config.json"))
        wandb.log_artifact(artifact)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    builder = ModelCachingModuleBuilder()
    run_id: str = generate_id()
    cfg: OfflineLabellingLMExperimentConfig = read_experiment_config(sys.argv[1],
                                                                     data_class=OfflineLabellingLMExperimentConfig,
                                                                     run_id=run_id)
    verify_cuda_is_available_if_needed()
    wandb.init(
        config=dataclasses.asdict(cfg),
        project="nc_latent_tod", entity=os.environ.get("WANDB_ENTITY", "kingb12"),
        name=cfg.wandb.run_name,
        notes=cfg.wandb.run_notes,
        group=cfg.wandb.run_group,
        tags=cfg.wandb.run_tags,
        id=run_id
    )
    main(cfg, builder=builder)
