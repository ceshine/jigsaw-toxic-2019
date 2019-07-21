import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, swish

from .dataset import AUX_COLUMNS
from .common import CACHE_DIR


class BertClassification(BertForSequenceClassification):
    def forward(self, input_ids, *args):
        return super().forward(input_ids, attention_mask=(input_ids > 0))


class CustomHead(nn.Module):
    def __init__(self, config, num_labels, dropout, start_layer):
        super().__init__()
        self.start_layer = start_layer
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(
            ((self.num_layers - start_layer + 1) // 2) *
            self.hidden_size, self.hidden_size
        )
        self.final_fc = nn.Linear(
            self.hidden_size, num_labels
        )
        self._init_weights()

    def _init_weights(self):
        for i, module in enumerate(self.children()):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, encoded_layers):
        extracted_layers = torch.cat([
            encoded_layers[i][:, 0] for i in range(self.start_layer, len(encoded_layers), 2)
        ], dim=1)
        out = self.dropout(self.pooler(extracted_layers))
        out = swish(out)
        return self.final_fc(out)


class BertCustomClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, dropout=0.1, head_start_layer=0):
        super(BertCustomClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
        self.head = CustomHead(config, num_labels, dropout,
                               start_layer=head_start_layer)

    def forward(self, input_ids, **kwargs):
        if 'attention_mask' in kwargs:
            del kwargs['attention_mask']
        encoded_layers, _ = self.bert(
            input_ids, attention_mask=(input_ids > 0),
            **kwargs
        )
        return self.head(encoded_layers)


class BertSplitHeadClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=15, split_point=6, dropout=0.1, head_start_layer=0):
        super(BertSplitHeadClassification, self).__init__(config)
        self.num_labels = num_labels
        self.split_point = split_point
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
        self.head_main = CustomHead(config, split_point, dropout,
                                    start_layer=head_start_layer)
        self.head_aux = CustomHead(config, num_labels - split_point, dropout,
                                   start_layer=head_start_layer)
        self.head = nn.ModuleList([self.head_main, self.head_aux])

    def forward(self, input_ids, **kwargs):
        if 'attention_mask' in kwargs:
            del kwargs['attention_mask']
        encoded_layers, _ = self.bert(
            input_ids, attention_mask=(input_ids > 0),
            output_all_encoded_layers=True, **kwargs
        )
        return torch.cat([
            self.head_main(encoded_layers),
            self.head_aux(encoded_layers)
        ], dim=1)


def get_model(bert_model):
    model = BertClassification.from_pretrained(
        bert_model, num_labels=len(AUX_COLUMNS))
    model.head = nn.ModuleList([model.classifier, model.bert.pooler])
    # if bert_model == "bert-base-uncased":
    #     print("Loading LM finetunned model...")
    #     model = BertSplitHeadClassification.from_pretrained(
    #         # model = BertCustomClassification.from_pretrained(
    #         str(CACHE_DIR / "bert_lm_uncased/"),
    #         num_labels=len(AUX_COLUMNS),
    #         split_point=6,
    #         dropout=0.1,
    #         head_start_layer=5
    #     )
    # else:
    # model = BertSplitHeadClassification.from_pretrained(
    # model = BertCustomClassification.from_pretrained(
    #     bert_model, num_labels=len(AUX_COLUMNS),
    #     dropout=0.1,
    #     # split_point=6,
    #     head_start_layer=5
    # )
    return model
