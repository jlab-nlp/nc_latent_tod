import re
from typing import List, Tuple, Optional

import torch
from transformers import StoppingCriteria, PreTrainedTokenizer


class StopSequencesStoppingCriteria:

    tokenizer: PreTrainedTokenizer
    stop_sequences: List[str]

    def __init__(self, stop_sequences: List[str], tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        # compile all stop sequences into a single regex, escaping regex special characters:
        self.stop_regex = re.compile("|".join(re.escape(seq) for seq in stop_sequences))

    def full_criteria(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, input_ids_length: int) -> bool:
        generated_ids: torch.LongTensor = input_ids[:, input_ids_length:] if len(input_ids.shape) == 2 else input_ids[
                                                                                                            input_ids_length:]
        outputs: List[str] = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, )
        # check that all strings have a regex match:
        should_stop: bool = all(self.stop_regex.search(output) for output in outputs)
        return should_stop

    def get_criteria(self, input_ids_length: int) -> StoppingCriteria:
        """
        Returns a StoppingCriteria (callable of input_ids, scores -> bool), which can be used to determine when to stop
        generation. Uses input_ids_length to determine which tokens to ignore when checking for stop sequences, checking
        only generated ids in each batch after this length.

        Usage:
        >>> criteria = StopSequencesStoppingCriteria(stop_sequences=["("], tokenizer=tokenizer)
        >>> generated_ids = self.model.generate(input_ids,  # normal input_ids tensor
                                            do_sample=True,  # should work in greedy as well, etc.
                                            num_return_sequences=4,
                                            max_new_tokens=256,
                                            # HF expects these in a list, so wrapping w/ []
                                            # make sure to call get_criteria() b.c. the stop_sequences must be enforced
                                            # in an input-dependent way
                                            stopping_criteria=[criteria.get_criteria(input_ids.shape[-1])])
        """

        return lambda input_ids, scores: self.full_criteria(input_ids, scores, input_ids_length)

    def trim_and_score_generations(self, generated_ids: torch.Tensor, scores: torch.Tensor, trim_trailing_whitespace: bool = True) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        if len(generated_ids.shape) == 1:
            generated_ids = generated_ids.unsqueeze(0)
            scores = scores.unsqueeze(0)
        for generated_sequence, sequence_scores in zip(generated_ids, scores, strict=True):
            generation: str = ""
            sequence_final_score: float = 0.0
            last_generation_length: int = 0
            for i in range(generated_ids.shape[-1]):
                generation = self.tokenizer.decode(generated_sequence[:i+1])
                possible_match: Optional[re.Match] = self.stop_regex.search(generation)
                if possible_match:
                    if possible_match.start() > last_generation_length:
                        # partial match: the last token includes but does not start with the stop_sequence:
                        # include the score of this token in the result tensor, but trim the generation to the
                        # start of the match
                        sequence_final_score = sequence_scores[:i+1].sum().item()
                    else:
                        # full match: the last token starts with the stop_sequence:
                        # trim the generation to the start of the match and do NOT include the last token's score
                        # in the result tensor
                        sequence_final_score = sequence_scores[:i].sum().item()
                    generation = generation[:possible_match.start()]
                    break
            results.append((generation.rstrip() if trim_trailing_whitespace else generation, sequence_final_score))
        return results
