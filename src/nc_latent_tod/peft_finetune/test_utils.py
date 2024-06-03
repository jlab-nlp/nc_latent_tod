import unittest
from typing import List

import torch
from transformers import AutoTokenizer

from nc_latent_tod.peft_finetune.utils import prompt_and_completion_to_inputs
from nc_latent_tod.utils.testing import test_suite


@test_suite("unit_build")
class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
        tokenizer.pad_token = tokenizer.eos_token
        cls.tokenizer = tokenizer

    def test_prompt_and_completion_to_inputs(self):
        prompt_a: str = "def foo(a, b, c):"
        prompt_b: str = "def add(a, b):\n    return a"
        completion_a: str = " return a + b + c"
        completion_b: str = " + b"
        prompt_a_ids: torch.Tensor = torch.tensor([1, 822, 7953, 29898, 29874, 29892, 289, 29892, 274, 1125])
        prompt_b_ids: torch.Tensor = torch.tensor([1, 822, 788, 29898, 29874, 29892, 289, 1125, 13, 1678, 736])
        completion_a_ids: torch.Tensor = torch.tensor([736, 263, 718, 289, 718, 274])
        completion_b_ids: torch.Tensor = torch.tensor([263, 718, 289])
        expected_input_ids: torch.Tensor = torch.stack([
            torch.cat([prompt_a_ids, completion_a_ids], dim=-1),
            torch.cat([torch.fill(torch.empty((2,)), 2), prompt_b_ids, completion_b_ids], dim=-1)
        ], dim=0).long()
        prompts: List[str] = [prompt_a, prompt_b]
        completions: List[str] = [completion_a, completion_b]

        # None: no masking, input_ids == label_ids
        batch = prompt_and_completion_to_inputs(self.tokenizer, {"prompt": prompts, "completion": completions})
        self.assertTrue(torch.equal(batch["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(batch["input_ids"], batch['labels']))
        # longer one has no padding (17)
        self.assertTrue(torch.equal(batch['input_ids'][0],
                                    torch.cat([prompt_a_ids, completion_a_ids], dim=-1)))
        # shorter one has padding (16)
        self.assertTrue(torch.equal(batch['input_ids'][1],
                                    torch.cat([torch.fill(torch.empty((2,)), 2), prompt_b_ids, completion_b_ids],
                                              dim=-1)))
        # padding: mask padding tokens
        batch = prompt_and_completion_to_inputs(self.tokenizer, {"prompt": prompts, "completion": completions},
                                                label_mask_mode="padding")
        expected_labels = expected_input_ids.clone()
        # sequences are 17, 16 in length, so we expect only 1 difference for the first padding token:
        expected_labels[1, 0] = -100
        expected_labels[1, 1] = -100
        self.assertTrue(torch.equal(expected_labels, batch['labels']))
        self.assertTrue(torch.equal(expected_input_ids, batch['input_ids']))

        # prompt: mask prompt tokens and padding tokens
        batch = prompt_and_completion_to_inputs(self.tokenizer, {"prompt": prompts, "completion": completions},
                                                label_mask_mode="prompt")
        expected_labels = torch.tensor([
            # we can have cases where the full text separates our 'prompt' from 'completion'. In this case, make sure
            # we don't mask the 'middle' token which belongs to both (-1).
            ([-100] * (prompt_a_ids.shape[0] - 1)) + prompt_a_ids[-1:].tolist() + completion_a_ids.tolist(),
            # second sequence is shorter, mask the 12 prompt tokens and 1 padding token:
            ([-100] * (prompt_b_ids.shape[0] + 2)) + completion_b_ids.tolist()
        ], dtype=torch.long)
        self.assertTrue(torch.equal(expected_labels, batch['labels']))
        self.assertTrue(torch.equal(expected_input_ids, batch['input_ids']))

    def test_singleton_batch(self):
        prompt = "def foo(a, b, c):"
        completion = "return a + b + c"
        for mode in [None, "padding", "prompt"]:
            # LMDataInstance as str values
            batch = prompt_and_completion_to_inputs(self.tokenizer, {"prompt": prompt, "completion": completion},
                                                    label_mask_mode=mode)
            self.assertLessEqual(batch['input_ids'].max(), 31_999)
            self.assertLessEqual(batch['labels'].max(), 31_999)
            # LMDataInstance as list[str] values
            batch_2 = prompt_and_completion_to_inputs(self.tokenizer, {"prompt": [prompt], "completion": [completion]},
                                                    label_mask_mode=mode)
            self.assertLessEqual(batch_2['input_ids'].max(), 31_999)
            self.assertLessEqual(batch_2['labels'].max(), 31_999)
            self.assertTrue(torch.equal(batch['input_ids'], batch_2['input_ids']))
            self.assertTrue(torch.equal(batch['labels'], batch_2['labels']))


if __name__ == '__main__':
    unittest.main()
