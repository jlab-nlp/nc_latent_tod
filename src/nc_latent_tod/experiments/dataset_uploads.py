import base64
import dataclasses
import hashlib
import json
import logging
import os
import pprint
import random
from typing import List, Any, Dict

import git
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import DatasetCard, HfApi
from tqdm import tqdm

from nc_latent_tod.data.utils import fill_all_states, get_hf_dataset_features, load_possibly_missing_dataset
from nc_latent_tod.data_types import DatasetTurnLog, DatasetTurn
from nc_latent_tod.experiments.config import OfflineLabellingLMExperimentConfig
from nc_latent_tod.experiments.manifest import ManifestEntry, ExperimentLogsManifest
from nc_latent_tod.peft_finetune.data import get_prompt_completion_dataset
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator, PromptMode
from nc_latent_tod.resources import read_resource
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.utils.dialogue_states import compute_delta, remove_blank_values
from nc_latent_tod.utils.general import group_by_dial_id_and_turn


def get_git_revision() -> str:
    try:
        repo: git.Repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError as e:
        logging.warning("running from outside of git repo, cannot get git revision")
        sha = "unknown"
    return sha


def generate_description_for_published_dataset(cfg: OfflineLabellingLMExperimentConfig) -> str:
    """
    Generates the README content for a published dataset from this self-labelling process on Huggingface
    """
    try:
        import wandb
        run_id: str = wandb.run.id
        run_name: str = wandb.run.name
    except BaseException as e:
        logging.warning("running outside of wandb, cannot get run id and name")
        run_id = "unknown"
        run_name = "unknown"

    template: str = read_resource("experiments/dataset_templates/hf_self_label_desc_template.md")
    return template.format(
        config=pprint.pformat(dataclasses.asdict(cfg)),
        script_path=__file__,
        git_revision=get_git_revision(),
        wandb_run_id=run_id,
        wandb_run_name=run_name,
    )


def generate_description_for_manifest_published_dataset(*, group_id: str, manifest_entries: List[ManifestEntry],
                                                        num_turns: int, num_dialogues: int, wandb_user: str = "kingb12") -> str:
    """
    Generates the README content for a published dataset from this self-labelling process on Huggingface
    """
    template: str = read_resource("experiments/dataset_templates/hf_self_label_desc_from_manifest_template.md")

    wandb_run_link_list: str = ""
    for entry in manifest_entries:
        wandb_run_link_list += f"- [{entry['run_id']}](https://wandb.ai/{wandb_user}/nc_latent_tod/runs/{entry['run_id']})" \
                               f" (`{entry['labelled_dataset_split_name']}`) \n"

    config_str: str = "{\"config\": \"not found\"}"
    run_id = manifest_entries[0]['run_id']
    try:
        import wandb
        run = wandb.Api().run(f"{wandb_user}/nc_latent_tod/{run_id}")
        config = run.config
        config_str = json.dumps(config, indent=4)
    except BaseException as e:
        logging.warning(f"unable to retriever run {run_id} to get config", e)

    manifest_entries_str = json.dumps([
        {
            'run_id': entry['run_id'],
            'dataset': entry['labelled_dataset_path_or_name'],
            'split': entry['labelled_dataset_split_name']
        } for entry in manifest_entries
    ], indent=4)
    return template.format(
        group_id=group_id,
        manifest_entries=manifest_entries_str,
        num_turns=num_turns,
        num_dialogues=num_dialogues,
        wandb_run_link_list=wandb_run_link_list,
        config=config_str

    )


def apply_self_labels(logs: List[DatasetTurnLog], schema: List[ServiceSchema]) -> List[DatasetTurn]:
    turns: List[DatasetTurn] = []
    index = group_by_dial_id_and_turn(logs, dialogue_id_key='dialogue_id', turn_id_key='turn_id')
    for log in tqdm(logs, "applying self-labels for dataset"):
        log: DatasetTurnLog
        pred_domains = log['pred_act_based_active_service_names'] or []
        for key in log['pred_belief_state']:
            pred_domains.append(key)
        pred_domains = list(set(pred_domains))
        last_pred_acts = index[log['dialogue_id']][log['turn_id'] - 1]['pred_system_response_acts'] if log[
                                                                                                           'turn_id'] > 0 else []
        turn: DatasetTurn = {
            "dialogue_id": log['dialogue_id'],
            "turn_id": log['turn_id'],
            "user_utterances": log['user_utterances'],
            "system_utterances": log['system_utterances'],
            "system_response": log['system_response'],
            # read only from predictions here!
            "system_response_acts": log['pred_system_response_acts'],
            "turn_slot_values": log['pred_delta_slot_values'],
            "slot_values": log['pred_belief_state'],
            "last_slot_values": log['pred_prior_context'],
            "domains": pred_domains,
            "last_system_response_acts": last_pred_acts
        }
        # replace real DST values with self-labelled predictions
        fill_all_states(turn, schema)
        assert compute_delta(
            remove_blank_values(turn['last_slot_values']),
            remove_blank_values(turn['slot_values'])
        ) == remove_blank_values(turn['turn_slot_values'])
        turns.append(turn)
    return turns


def build_and_upload_self_labeled_dataset(cfg: OfflineLabellingLMExperimentConfig, running_log: List[DatasetTurnLog],
                                          schema: List[ServiceSchema]):
    running_log = apply_self_labels(running_log, schema)

    # instantiate as dataset (to be a split in a larger one)
    dataset: Dataset = Dataset.from_list(running_log, features=get_hf_dataset_features(schema))
    # come up with a name for the dataset and split
    dataset_name_or_path: str = cfg.publish_labelled_dataset_as.path_or_name
    split_name: str = cfg.publish_labelled_dataset_as.split_name or cfg.data.eval_set_split_name
    dataset_dict: DatasetDict = load_possibly_missing_dataset(dataset_name_or_path) or DatasetDict({
        split_name: dataset
    })
    dataset_dict[split_name] = dataset
    # finally, upload
    if cfg.publish_labelled_dataset_as.push_to_hub:
        dataset_dict.push_to_hub(dataset_name_or_path, private=True)
        dataset_card = DatasetCard(content=generate_description_for_published_dataset(cfg))
        HfApi().upload_file(
            path_or_fileobj=str(dataset_card).encode(),
            path_in_repo="README.md",
            repo_id=dataset_name_or_path,
            repo_type="dataset",
        )


def short_hash(jsonable_obj: Any, length: int = 12) -> str:
    json_string = json.dumps(jsonable_obj, sort_keys=True)
    # Generate a SHA-256 hash of the JSON string
    hash_obj = hashlib.sha256(json_string.encode())
    hash_digest = hash_obj.digest()
    # Encode the hash using base64 and truncate it
    return base64.urlsafe_b64encode(hash_digest)[:length].decode('utf-8')


def upload_from_manifest_group(manifest: ExperimentLogsManifest, group_id: str,
                               schema: List[ServiceSchema],
                               dataset_name_or_path: str = None,
                               dataset_split_name: str = "train",
                               hf_username: str = "Brendan",
                               is_private_upload: bool = True) -> str:
    """
    A manifest group organizes a set of logs that were generated in a batched experiment run. I.e, when offline
    labelling, we usually label dialogues ~50 at a time, and add them to a manifest group. Given a manifest group
    identifier, this method uploads all self-labels predicted in the logs, and generates a dataset from them to publish.

    :param manifest: a pointer to a Manifest, which contains different named/id'd groups
    :param group_id: the name/id of the group to upload
    :param schema: schema matching all logs in that group
    :param dataset_name_or_path: the output dataset name or path
    :param dataset_split_name: a name for the split of the dataset, defaults to train.

    :return: the name of the now uploaded dataset
    """
    turns: List[DatasetTurn] = apply_self_labels(manifest.get_logs_from_group(group_id), schema)
    num_dialogues: int = len(set([turn['dialogue_id'] for turn in turns]))
    num_turns: int = len(turns)
    manifest_entries: List[ManifestEntry] = manifest.read_group(group_id)
    description: str = generate_description_for_manifest_published_dataset(
        group_id=group_id,
        manifest_entries=manifest_entries,
        num_dialogues=num_dialogues,
        num_turns=num_turns
    )
    # generate a short hash string from manifest_entries, for uniqueness in name
    hash_str: str = short_hash(manifest_entries).replace('-', 'x')
    dataset_name_or_path = dataset_name_or_path or f"{hf_username}/manifest_self_labelled_{group_id}_{num_dialogues}_{hash_str}"
    dataset_dict: DatasetDict = DatasetDict({
        dataset_split_name: Dataset.from_list(turns, features=get_hf_dataset_features(schema))
    })
    dataset_dict.push_to_hub(dataset_name_or_path, private=is_private_upload)
    logging.info(f"Pushed dataset to {dataset_name_or_path}")
    dataset_card = DatasetCard(content=description)
    HfApi().upload_file(
        path_or_fileobj=str(dataset_card).encode(),
        path_in_repo="README.md",
        repo_id=dataset_name_or_path,
        repo_type="dataset",
    )
    return dataset_name_or_path


def generate_description_for_premixed_dataset(*, source_path_or_name: str, source_split: str, target_path_or_name: str,
                                              target_split: str, prompt_generator: AbstractPromptGenerator,
                                              modes: Dict[PromptMode, float]) -> str:
    template: str = read_resource("experiments/dataset_templates/hf_self_label_premixed.md")
    pretty_mode_table_strings: List[str] = [
        f"| `{mode}` | {proportion} |" for mode, proportion in sorted(modes.items(), key=lambda x: x[0])
    ]
    pretty_mode_table: str = "\n".join(pretty_mode_table_strings)
    return template.format(
        source_path_or_name=source_path_or_name,
        source_split=source_split,
        prompt_generator_class=prompt_generator.__class__.__name__,
        mixture_table_rows=pretty_mode_table,
        git_revision=get_git_revision(),
    )


def upload_premixed_dataset(*, source_path_or_name: str, source_split: str, target_path_or_name: str, target_split: str,
                            prompt_generator: AbstractPromptGenerator,
                            modes: Dict[PromptMode, float]) -> Dataset:
    dataset_card = DatasetCard(content=generate_description_for_premixed_dataset(
        source_path_or_name=source_path_or_name,
        source_split=source_split,
        target_path_or_name=target_path_or_name,
        target_split=target_split,
        prompt_generator=prompt_generator,
        modes=modes
    ))
    source_dataset: Dataset = load_dataset(source_path_or_name, split=source_split)
    dataset: Dataset = get_prompt_completion_dataset(
        prompt_generator=prompt_generator,
        dataset=source_dataset, modes=modes,
        num_proc=min(os.cpu_count(), 8)
    )
    # Note: target wraps the dataset, but caller just wants the one split (presumably)
    target_dataset: DatasetDict = DatasetDict({
        target_split: dataset
    })
    target_dataset.push_to_hub(target_path_or_name, private=True)
    HfApi().upload_file(
        path_or_fileobj=str(dataset_card).encode(),
        path_in_repo="README.md",
        repo_id=target_path_or_name,
        repo_type="dataset",
    )
    return dataset


def create_train_valid_dialogue_split(dataset_name: str, split_name: str = "train", num_dialogues: int = 64):
    dataset: DatasetDict = load_dataset(dataset_name)
    all_dialogue_ids: List[str] = dataset[split_name].unique('dialogue_id')

    rng = random.Random(42)  # local seed so we don't rely on any global one
    rng.shuffle(all_dialogue_ids)
    valid_dialogue_ids = all_dialogue_ids[:num_dialogues]
    train_dialogue_ids = all_dialogue_ids[num_dialogues:]
    valid_dataset = dataset[split_name].filter(lambda example: example['dialogue_id'] in valid_dialogue_ids)
    train_dataset = dataset[split_name].filter(lambda example: example['dialogue_id'] in train_dialogue_ids)
    # verify no overlap
    assert set(valid_dataset.unique('dialogue_id')).isdisjoint(set(train_dataset.unique('dialogue_id')))
    dataset[f'train_minus_{num_dialogues}'] = train_dataset
    dataset[f'valid_{num_dialogues}'] = valid_dataset
    dataset.push_to_hub(dataset_name, private=True)
