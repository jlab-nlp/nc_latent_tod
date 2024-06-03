import logging
from typing import TypedDict, Union, List, Dict, Optional

from datasets import DatasetDict, Dataset, concatenate_datasets

from nc_latent_tod.data_types import DatasetTurn
from nc_latent_tod.prompting.abstract_prompt_generator import AbstractPromptGenerator, PromptMode
from nc_latent_tod.utils.general import group_by_dial_id_and_turn


class LMDataInstance(TypedDict):
    """
    We create a training dataset with each instance being a simple dict with keys "prompt" and "completion"
    """
    prompt: str
    completion: str


def get_dataset_proportion(dataset: Dataset, proportion: float, shuffle: bool = True) -> Dataset:
    # use seed for repeatability
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    # get indices
    if proportion < 1:
        max_index: int = int(len(dataset) * proportion)
        return dataset.select(range(max_index))
    elif proportion == 1:
        return dataset
    else:
        indices = []
        while proportion > 1:
            # keep adding full dataset
            indices.extend(range(len(dataset)))
            proportion -= 1
        assert proportion >= 0
        if proportion > 0:
            # handle remainder
            max_index: int = int(len(dataset) * proportion)
            indices.extend(range(max_index))
        return dataset.select(indices)


def get_instance(prompt_generator: AbstractPromptGenerator, turn: DatasetTurn, mode: PromptMode,
                 dialogue_turn_lookup: Optional[Dict[str, List[DatasetTurn]]],
                 context_turns: int = 0) -> LMDataInstance:
    # if we specify some number of context turns, pick past turns from this dialogue as examples
    examples: List[DatasetTurn] = []
    if context_turns:
        for i in range(context_turns, 0, -1):
            example_turn_id: int = turn['turn_id'] - i
            if example_turn_id >= 0:
                example_turn = dialogue_turn_lookup[turn['dialogue_id']][turn['turn_id'] - i]
                if example_turn:
                    # double check this is the right turn
                    assert example_turn['turn_id'] == example_turn_id
                    assert example_turn['dialogue_id'] == turn['dialogue_id']
                    examples.append(example_turn)
    prompt, completion = prompt_generator.get_finetuning_prompt_and_completion(turn, mode=mode, examples=examples)
    return {
        "prompt": prompt,
        "completion": completion
    }


def get_prompt_completion_dataset(prompt_generator: AbstractPromptGenerator, dataset: Union[DatasetDict, Dataset], modes: Dict[PromptMode, float],
                                  shuffle: bool = True, num_proc: int = 16, context_turns: int = 0) -> Dataset:
    task_datasets: List[Dataset] = []
    dialogue_turn_lookup = group_by_dial_id_and_turn(dataset, dialogue_id_key='dialogue_id', turn_id_key='turn_id')
    for mode, proportion in modes.items():
        logging.info(f"Generating prompt-completion dataset for mode {mode} ({proportion}x)")
        dataset_mix_part = get_dataset_proportion(dataset, proportion=proportion, shuffle=shuffle)
        task_data = dataset_mix_part.map(
            # maps each turn to a dict with 'prompt' and 'completion'
            lambda turn: get_instance(turn=turn, mode=mode, prompt_generator=prompt_generator,
                                      context_turns=context_turns, dialogue_turn_lookup=dialogue_turn_lookup),
            num_proc=num_proc
        )
        task_datasets.append(task_data)
    new_dataset = concatenate_datasets(task_datasets)
    if shuffle:
        new_dataset = new_dataset.shuffle(seed=42)
    return new_dataset
