import dataclasses
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, List, Union, Optional, Dict, Tuple

import datasets
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from nc_latent_tod.experiments.dataset_uploads import upload_premixed_dataset
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, AutoTokenizer, DataCollatorForTokenClassification
from wandb.sdk.lib.runid import generate_id

from nc_latent_tod.experiments.utils import scale_batch_size_to_memory
from nc_latent_tod.clients.abstract_hf_lm_client import AbstractHFLMClient
from nc_latent_tod.db.abstract_db import AbstractDB
from nc_latent_tod.delex.fuzzy_regex_delexer import FuzzyRegexDelexer
from nc_latent_tod.experiments.batch_client_lm_module import BatchLMClientDSTModule, BatchLMClientActTagModule, \
    BatchLMClientPolicyModule, BatchLMClientResponseGenModule
from nc_latent_tod.experiments.config import LMModelConfig, OutputConfigDC, \
    WandbConfigDC, DataConfigDC, SchemaConfigDC, FullMultiWOZSchemaConfig, BatchModuleConfig
from nc_latent_tod.experiments.offline_labelling_experiment import OfflineLabellingExperiment
from nc_latent_tod.experiments.online_e2e_experiment import OnlineE2EExperiment
from nc_latent_tod.experiments.utils import get_client_from_lm_model_config, verify_cuda_is_available_if_needed, \
    load_datasets_from_cfg, get_db_from_schema_config, read_experiment_config
from nc_latent_tod.kwargs_prompt.simple_ft_prompt import SimpleFTKwargsPromptGenerator
from nc_latent_tod.normalization.abstract_normalizer import AbstractNormalizer
from nc_latent_tod.normalization.schema_normalizer import SchemaNormalizer
from nc_latent_tod.ontology.abstract_ontology import AbstractDBOntology
from nc_latent_tod.peft_finetune.data import get_prompt_completion_dataset
from nc_latent_tod.peft_finetune.utils import prompt_and_completion_to_inputs, LabelMaskMode
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator, PromptMode
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.utils.callbacks import TFLOPSCallback
from nc_latent_tod.utils.general import write_json, FunctionsAsNamesEncoder
from nc_latent_tod.utils.turn_logger import WandbArtifactViewerMixin


@dataclass
class TrainingConfig:
    label_mask_mode: LabelMaskMode = None
    save_steps: Optional[int] = 100
    max_steps: Optional[int] = 200
    eval_steps: Optional[int] = 100
    eval_delay: Optional[int] = 0
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    num_eval_dialogues_to_subsample: int = -1  # default to evaluating with the complete evaluation set
    memory_metrics: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    max_sequence_length: Optional[int] = None
    evaluate_on_start: bool = True
    optim: str = "adamw_torch"
    torch_compile: bool = False
    use_peft: bool = False
    lora_config: LoraConfig = LoraConfig()


@dataclass
class TrainingDataConfigDC(DataConfigDC):
    # adding dict for task weighting, trying to accomodate a list format if possible
    prompt_modes: Union[List[PromptMode], Dict[PromptMode, float]] = None
    num_dialogues: int = -1
    load_premixed_and_tokenized_dataset: Optional[str] = None
    upload_premixed_dataset: bool = False


@dataclass
class MultitaskFineTuneConfig:
    model: LMModelConfig
    data: TrainingDataConfigDC
    dst: Optional[BatchModuleConfig] = None
    act_tag: Optional[BatchModuleConfig] = None
    policy: Optional[BatchModuleConfig] = None
    response_gen: Optional[BatchModuleConfig] = None
    eval_offline_label: bool = False
    eval_online_e2e: bool = False
    output: OutputConfigDC = OutputConfigDC()
    wandb: WandbConfigDC = WandbConfigDC()
    schema: SchemaConfigDC = FullMultiWOZSchemaConfig
    training: TrainingConfig = TrainingConfig()
    profile: bool = False  # deprecated
    experiment_outer_batch_size: int = 0


class MultitaskFinetuneTrainer(Trainer):
    prompt_generator: AbstractPromptGenerator
    cfg: MultitaskFineTuneConfig
    schema_loader: Callable[[], List[ServiceSchema]]
    ontology: AbstractDBOntology
    normalizer: AbstractNormalizer

    def __init__(self,
                 client: AbstractHFLMClient = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 prompt_generator: AbstractPromptGenerator = None,
                 cfg: MultitaskFineTuneConfig = None,
                 schema_loader: Callable[[], List[ServiceSchema]] = None,
                 ontology: AbstractDBOntology = None,
                 normalizer: AbstractNormalizer = None):
        super().__init__(client.model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.normalizer = normalizer
        self.ontology = ontology
        self.prompt_generator = prompt_generator
        self.cfg = cfg
        self.schema_loader = schema_loader

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            num_eval_dialogues_to_subsample: Optional[int] = None,
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        # override only if None: this allows us to disable sub-sampling by setting to -1 in the call to evaluate()
        if num_eval_dialogues_to_subsample is None:
            num_eval_dialogues_to_subsample = self.cfg.training.num_eval_dialogues_to_subsample
        eval_dataset: Dataset = eval_dataset or self.eval_dataset

        eval_model = self._wrap_model(self.model, training=False).eval()

        if num_eval_dialogues_to_subsample > 0:
            # sub-sample K dialogues before evaluating
            dialogue_ids = random.sample(eval_dataset.unique('dialogue_id'), k=num_eval_dialogues_to_subsample)
            eval_dataset = eval_dataset.filter(lambda x: x['dialogue_id'] in dialogue_ids)

        if self.cfg.eval_offline_label:
            assert self.cfg.dst is not None, "need to specify a DST config to evaluate offline labelling"
            assert self.cfg.act_tag is not None, "need to specify an act tagging config to evaluate offline labelling"
            defaults: BatchModuleConfig = BatchModuleConfig()
            defaults.model = dataclasses.replace(defaults.model, **vars(self.cfg.model))
            dst_cfg: BatchModuleConfig = dataclasses.replace(defaults, **vars(self.cfg.dst))
            act_tag_cfg: BatchModuleConfig = dataclasses.replace(defaults, **vars(self.cfg.act_tag))

            batch_size: int = act_tag_cfg.model.batch_size or self.args.per_device_eval_batch_size
            batch_size = scale_batch_size_to_memory(batch_size)
            experiment_outer_batch_size: int = self.cfg.experiment_outer_batch_size or batch_size

            experiment: OfflineLabellingExperiment = OfflineLabellingExperiment(
                dst_module=BatchLMClientDSTModule(
                    prompt_generator=self.prompt_generator,
                    # override the client model with the one we're training
                    client=get_client_from_lm_model_config(dst_cfg.model, model=eval_model, max_batch_size=batch_size),
                    generation_cfg=dst_cfg.generation_cfg,
                ),
                act_module=BatchLMClientActTagModule(
                    prompt_generator=self.prompt_generator,
                    # override the client model with the one we're training
                    client=get_client_from_lm_model_config(act_tag_cfg.model, model=eval_model, max_batch_size=batch_size),
                    generation_cfg=act_tag_cfg.generation_cfg,
                ),
                test_set=eval_dataset,
                output_dir=self.cfg.output.output_dir,
                schema_loader=self.schema_loader,
                ontology=self.ontology,
                normalizer=self.normalizer,
                batch_size=experiment_outer_batch_size,
                viewers=[
                    DSTTextFileLogViewer(
                        log_file_path=os.path.join(self.cfg.output.output_dir, f"dst_logs_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    ),
                    ActTaggingTextFileLogViewer(
                        log_file_path=os.path.join(self.cfg.output.output_dir,
                                                   f"act_tagging_logs_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    ),
                ],
            )
            logging.info("Running offline labelling experiment")
            logs, stats = experiment.run()
            # upload viewer artifacts:
            for viewer in experiment.turn_logger.viewers:
                if isinstance(viewer, WandbArtifactViewerMixin):
                    try:
                        artifact: wandb.Artifact = viewer.to_wandb_artifact()
                        wandb.log_artifact(artifact)
                    except BaseException as e:
                        # continue, don't stop training for this
                        logging.error(f"Failed to upload viewer artifact: {e}")
            artifact: wandb.Artifact = wandb.Artifact(f"{wandb.run.id}_{self.state.global_step}", type="run_output")
            if os.path.exists(os.path.join(self.cfg.output.output_dir, f"running_log.json")):
                # we don't write for logs < 20 items, so this may not exist
                artifact.add_file(os.path.join(self.cfg.output.output_dir, f"running_log.json"))
                artifact.add_file(os.path.join(self.cfg.output.output_dir, f"exp_config.json"))
            wandb.log_artifact(artifact)
            result.update({f"{metric_key_prefix}_offline_label_{metric}": value for metric, value in stats.items()})

        if self.cfg.eval_online_e2e:
            assert self.cfg.dst is not None, "need to specify a DST config to evaluate E2E"
            assert self.cfg.policy is not None, "need to specify a policy config to evaluate E2E"
            assert self.cfg.response_gen is not None, "need to specify a response generation config to evaluate E2E"
            defaults: BatchModuleConfig = BatchModuleConfig()
            dst_cfg: BatchModuleConfig = dataclasses.replace(defaults, **vars(self.cfg.dst))
            policy_cfg: BatchModuleConfig = dataclasses.replace(defaults, **vars(self.cfg.policy))
            resp_gen_cfg: BatchModuleConfig = dataclasses.replace(defaults, **vars(self.cfg.response_gen))

            batch_size: int = dst_cfg.model.batch_size or self.args.per_device_eval_batch_size
            batch_size = scale_batch_size_to_memory(batch_size)

            experiment: OnlineE2EExperiment = OnlineE2EExperiment(
                dst_module=BatchLMClientDSTModule(
                    prompt_generator=self.prompt_generator,
                    # override the client model with the one we're training
                    client=get_client_from_lm_model_config(dst_cfg.model, model=eval_model, max_batch_size=batch_size),
                    generation_cfg=dst_cfg.generation_cfg,
                ),
                policy_module=BatchLMClientPolicyModule(
                    prompt_generator=self.prompt_generator,
                    # override the client model with the one we're training
                    client=get_client_from_lm_model_config(policy_cfg.model, model=eval_model, max_batch_size=batch_size),
                    generation_cfg=policy_cfg.generation_cfg,
                ),
                response_gen_module=BatchLMClientResponseGenModule(
                    prompt_generator=self.prompt_generator,
                    # override the client model with the one we're training
                    client=get_client_from_lm_model_config(resp_gen_cfg.model, model=eval_model, max_batch_size=batch_size),
                    generation_cfg=resp_gen_cfg.generation_cfg,
                ),
                test_set=eval_dataset,
                output_dir=self.cfg.output.output_dir,
                schema_loader=self.schema_loader,
                ontology=self.ontology,
                normalizer=self.normalizer,
                batch_size=batch_size,
                delexer=FuzzyRegexDelexer(),
                viewers=[
                    DSTTextFileLogViewer(
                        log_file_path=os.path.join(self.cfg.output.output_dir, f"dst_log_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    ),
                    PolicyTextFileLogViewer(
                        log_file_path=os.path.join(self.cfg.output.output_dir, f"pol_log_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    ),
                    ResponseGenTextFileLogViewer(
                        log_file_path=os.path.join(self.cfg.output.output_dir, f"rg_log_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    ),
                    FullDialogueViewer(
                        schema=self.prompt_generator.schema,
                        log_file_path=os.path.join(self.cfg.output.output_dir, f"full_{self.state.global_step}.txt"),
                        step=self.state.global_step
                    )
                ]
            )
            logging.info("Running online e2e experiment")
            logs, stats = experiment.run()
            # upload viewer artifacts:
            for viewer in experiment.turn_logger.viewers:
                if isinstance(viewer, WandbArtifactViewerMixin):
                    try:
                        artifact: wandb.Artifact = viewer.to_wandb_artifact()
                        wandb.log_artifact(artifact)
                    except BaseException as e:
                        # continue, don't stop training for this
                        logging.error(f"Failed to upload viewer artifact: {e}")

            artifact: wandb.Artifact = wandb.Artifact(f"{wandb.run.name}_{self.state.global_step}", type="run_output")
            if os.path.exists(os.path.join(self.cfg.output.output_dir, f"running_log.json")):
                # we don't write for logs < 20 items, so this may not exist
                artifact.add_file(os.path.join(self.cfg.output.output_dir, f"running_log.json"))
                artifact.add_file(os.path.join(self.cfg.output.output_dir, f"exp_config.json"))
            wandb.log_artifact(artifact)
            result.update({f"{metric_key_prefix}_online_e2e_{metric}": value for metric, value in stats.items()})
            result[f"{metric_key_prefix}_online_e2e_total_success"] = stats['success']['success']['total']
        if not self.cfg.eval_offline_label and not self.cfg.eval_online_e2e:
            logging.warning("Not evaluating anything!")
        return result


def prepare_peft_model(model: PreTrainedModel, lora_config: LoraConfig, adapter_name: str = "default") -> PeftModel:
    # some model preparation work done by `peft`
    model = prepare_model_for_kbit_training(model)

    # For our parameter efficient tuning method, we'll use LoRA
    defaults = LoraConfig(
        lora_dropout=.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"]
    )
    lora_config = dataclasses.replace(defaults, **vars(lora_config))

    # get a peft model based on our config and base model
    peft_model: PeftModel = get_peft_model(model=model, peft_config=lora_config, adapter_name=adapter_name)
    return peft_model


def get_prompt_mode_weights(prompt_modes: Union[List[PromptMode], Dict[PromptMode, float]]) -> Dict[PromptMode, float]:
    if isinstance(prompt_modes, list):
        return {mode: 1.0 for mode in prompt_modes}
    return prompt_modes



def main(cfg: MultitaskFineTuneConfig):
    assert cfg.output.output_dir, "need to specify an output_dir. Callers of main() should specify a default"
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    # write out this experiment's configuration
    write_json(dataclasses.asdict(cfg),
               os.path.join(cfg.output.output_dir, "exp_config.json"),
               # serialize functions by their names, we don't need to de-serialize them
               indent=4, cls=FunctionsAsNamesEncoder)

    # Create a TFLOPs Callback which logs to wandb
    tflops_callback: TFLOPSCallback = TFLOPSCallback(logging_callback=wandb.log)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    train_set, eval_set = load_datasets_from_cfg(cfg.data)

    if cfg.data.num_dialogues > 0:
        local_random = random.Random(42)
        train_dialogue_ids = local_random.choices(train_set.unique('dialogue_id'), k=cfg.data.num_dialogues)
        logging.info(f"Down-sampling {cfg.data.num_dialogues} dialogues for training: {', '.join(train_dialogue_ids)}")
        train_set = train_set.filter(lambda x: x['dialogue_id'] in train_dialogue_ids)

    # config includes lazy loading provider functions
    schema: List[ServiceSchema] = cfg.schema.schema_loader()
    ontology: AbstractDBOntology = cfg.schema.ontology()
    db: AbstractDB = get_db_from_schema_config(cfg.schema)
    normalizer: AbstractNormalizer = SchemaNormalizer(schema=schema)
    prompt_generator: SimpleFTKwargsPromptGenerator = SimpleFTKwargsPromptGenerator(schema=schema, db=db,
                                                                                    ontology=ontology)

    prompt_mode_weights: Dict[PromptMode, float] = get_prompt_mode_weights(cfg.data.prompt_modes)

    # for whatever reason, starcoder's tokenizer doesn't specify its pad token, and if we don't set it, then when we go
    # to pad batches in the data collator (DataCollatorWithPadding, default from Trainer) it breaks. Setting here for
    # use anywhere we pad. See: https://huggingface.co/bigcode/starcoder/discussions/67
    if not hasattr(tokenizer, 'pad_token') or not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.data.load_premixed_and_tokenized_dataset:
        tokenized_train_set: Dataset = datasets.load_from_disk(cfg.data.load_premixed_and_tokenized_dataset)
    else:
        if cfg.data.upload_premixed_dataset:
            promptable_train_set: Dataset = upload_premixed_dataset(
                source_path_or_name=cfg.data.train_set_path_or_name,
                source_split=cfg.data.train_set_split_name,
                target_path_or_name=f"{cfg.data.train_set_path_or_name}_premixed_{run_id}",
                target_split=cfg.data.train_set_split_name,
                prompt_generator=prompt_generator,
                modes=prompt_mode_weights
            )
        else:
            # Just generate, don't upload
            promptable_train_set: Dataset = get_prompt_completion_dataset(
                prompt_generator=prompt_generator,
                dataset=train_set,
                modes=prompt_mode_weights
            )
        # not calling batched! batched adds padding, which we'll add in the Collator. Otherwise, we re-pad
        tokenized_train_set = promptable_train_set.map(
            lambda item: prompt_and_completion_to_inputs(tokenizer, item, label_mask_mode=cfg.training.label_mask_mode),
            num_proc=min(os.cpu_count(), 8),
        )

    # if we've configured a max sequence length, drop other examples
    if cfg.training.max_sequence_length:
        tokenized_train_set = tokenized_train_set.filter(
            lambda item: len(item['input_ids']) <= cfg.training.max_sequence_length, num_proc=min(os.cpu_count(), 8)
        )
    # In training, I think the choice of padding is mostly irrelevant, but we'll use left padding to best support batch
    # inference later on. Right side padding makes sequences more uniform, but left will is needed for batch inference.
    tokenizer.padding_side = 'left'
    tokenized_train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    client: AbstractHFLMClient = get_client_from_lm_model_config(cfg.model, load_in_8bit=False)
    client.model = client.model.train()
    if cfg.training.use_peft:
        client.model = prepare_peft_model(client.model, cfg.training.lora_config)
    training_args = TrainingArguments(
        output_dir=cfg.output.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=cfg.training.max_steps,
        eval_steps=cfg.training.eval_steps,
        eval_delay=cfg.training.eval_delay,
        save_steps=cfg.training.save_steps,
        save_total_limit=3,
        save_only_model=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        logging_steps=50,
        # We're optimizing training speed but in a real setup you can increase eval batch size beyond train batch size
        per_device_train_batch_size=scale_batch_size_to_memory(cfg.training.per_device_train_batch_size),
        per_device_eval_batch_size=scale_batch_size_to_memory(cfg.training.per_device_eval_batch_size),
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        weight_decay=0.05,
        metric_for_best_model="offline_label_jga" if cfg.eval_offline_label else "online_e2e_total_success",
        report_to=["wandb"],
        skip_memory_metrics=not cfg.training.memory_metrics,
        # implementing @karpathy's simple speed-ups for the dataloader. If using k8s, make sure cpu requests > this val
        dataloader_num_workers=cfg.training.num_workers,
        dataloader_pin_memory=cfg.training.pin_memory,
        optim=cfg.training.optim,
        torch_compile=cfg.training.torch_compile,
    )
    all_callbacks: List[TrainerCallback] = [tflops_callback]
    trainer = MultitaskFinetuneTrainer(
        client=client,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        prompt_generator=prompt_generator,
        cfg=cfg,
        schema_loader=cfg.schema.schema_loader,
        ontology=cfg.schema.ontology,
        normalizer=normalizer,
        callbacks=all_callbacks,
        data_collator=DataCollatorForTokenClassification(tokenizer, padding=True),
    )
    training_callable: Callable[[], None] = trainer.train


    if cfg.training.evaluate_on_start:
        trainer.evaluate()
    training_callable()
    # final evaluation
    trainer.evaluate(eval_dataset=eval_set)

    if cfg.output.on_end_copy_to_dir:
        logging.info(f"Copying outputs to {cfg.output.on_end_copy_to_dir}")
        os.system(f"cp -r {cfg.output.output_dir} {cfg.output.on_end_copy_to_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_id: str = generate_id()
    logging.info(f"Starting run {run_id} with config {sys.argv[1]}")
    cfg: MultitaskFineTuneConfig = read_experiment_config(
        config_path=sys.argv[1],
        data_class=MultitaskFineTuneConfig,
        run_id=run_id
    )

    verify_cuda_is_available_if_needed()
    wandb.init(
        config=dataclasses.asdict(cfg),
        project="nc_latent_tod", entity=os.environ.get("WANDB_ENTITY", "kingb12"),
        name=cfg.wandb.run_name,
        notes=cfg.wandb.run_notes,
        group=cfg.wandb.run_group,
        tags=cfg.wandb.run_tags,
        id=run_id,
    )
    main(cfg)
