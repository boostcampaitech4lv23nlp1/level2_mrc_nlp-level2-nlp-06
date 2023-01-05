import torch.nn as nn
from transformers import (
    AutoModel,
    BertModel,
    BigBirdModel,
    BertPreTrainedModel,
    BigBirdPreTrainedModel,
)


class DenseRetriever(nn.Module):
    """Encoder creating the dense representation

    Attributes:
        self.model (): The model to create the dense representation
    """
    def __init__(self, config):
        super(DenseRetriever, self).__init__()

        self.config = config
        self.model = AutoModel.from_pretrained(
            config["model_name_or_path"]
        )


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output
