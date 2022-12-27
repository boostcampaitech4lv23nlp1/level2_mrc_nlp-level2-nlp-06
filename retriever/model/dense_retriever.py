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


class BertEncoder(BertPreTrainedModel):
    """BERT-based encoder creating the dense representation

    You can use the pre-trained BERT in HuggingFace running the following example:
        encoder = BertEncoder.from_pretrained('klue/bert-base')

    Attributes:
        self.bert (transformers.models.bert.modeling_bert.BertModel):
            The model to create the dense representation

            CAUTION! DO NOT CHANGE THE NAME OF THIS ATTRIBUTE.
            It may initialize the weights randomly without loading the pre-trained weights.
    """


    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class BigBirdEncoder(BigBirdPreTrainedModel):
    """BERT-based encoder for creating the dense representation of questions and passages

    You can use the pre-trained BERT in HuggingFace running the following example:
        encoder = BigBirdEncoder.from_pretrained('monologg/kobigbird-bert-base')

    Attributes:
        self.bert (transformers.models.big_bird.modeling_big_bird.BigBirdModel):
            The model to create the dense representation

            CAUTION! DO NOT CHANGE THE NAME OF THIS ATTRIBUTE.
            It may initialize the weights randomly without loading the pre-trained weights.
    """


    def __init__(self, config):
        super(BigBirdEncoder, self).__init__(config)

        self.bert = BigBirdModel(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output
