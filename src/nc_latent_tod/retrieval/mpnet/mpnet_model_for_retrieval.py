from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import MPNetModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

"""
Implements a dense retriever using mpnet and mean pooling. Identical to the SentenceTransformers module, but the purpose
is to write my own huggingface training loop (which will hopefully be faster than the one implemented in RefPyDST)
"""


class MPNetModelForRetrieval(MPNetModel):

    def __init__(self, config, **kwargs):
        # block the pooling layer instantiation since we won't use it and will do it ourselves
        super().__init__(config, add_pooling_layer=False)

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def normalize(embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], BaseModelOutputWithPooling]:
        if return_dict or (return_dict is None and self.config.use_return_dict):
            outputs = super().forward(input_ids, attention_mask=attention_mask, return_dict=return_dict, **kwargs)
            outputs['pooled_output'] = self.normalize(self.mean_pooling(outputs['last_hidden_state'], attention_mask))
            return outputs
        else:
            last_hidden_state: torch.Tensor
            last_hidden_state, _ = super().forward(input_ids, attention_mask=attention_mask, return_dict=return_dict, **kwargs)
            return (last_hidden_state, self.normalize(self.mean_pooling(last_hidden_state, attention_mask)))
