
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import GPT2Model
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel
from pytorch_pretrained_bert.modeling import swish

from .dataset import AUX_COLUMNS

SPECIAL_TOKENS = ["<pad>", "<cls>"]


class AttentionHead(nn.Module):
    def __init__(self, config, n_class=8):
        super().__init__()
        self.linear = nn.Sequential(
            # nn.BatchNorm1d(config.n_embd * 2),
            # nn.Dropout(clf_dropout),
            nn.Linear(config.n_embd, n_class)
        )
        self.attention_mapping = nn.Linear(config.n_embd, 128, bias=False)
        self.attention = nn.Parameter(torch.Tensor(128))
        nn.init.uniform_(self.attention, -.2, .2)
        for i, module in enumerate(self.linear):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, hidden_states):
        attentions = F.softmax(
            torch.bmm(
                torch.tanh(self.attention_mapping(hidden_states)),
                self.attention[None, :, None].repeat(
                    hidden_states.size(0), 1, 1)
            ), dim=1
        ) * (input_ids != 50256).unsqueeze(2).float()
        # print(torch.min(hidden_states), torch.max(
        #     hidden_states), torch.mean(hidden_states))
        # print(attentions[-1, :, 0])
        weights = attentions.sum(dim=1, keepdim=True)
        # re-normalize
        attentions = attentions / weights
        pooled = torch.sum(hidden_states * attentions, dim=1)
        logits = self.linear(pooled)
        return logits


class PoolingHead(nn.Module):
    def __init__(self, config, n_class=8):
        super().__init__()
        self.linear = nn.Sequential(
            # nn.BatchNorm1d(config.n_embd * 2),
            # nn.Dropout(clf_dropout),
            nn.Linear(config.n_embd, n_class)
        )
        for i, module in enumerate(self.linear):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    assert self.linear[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, hidden_states):
        mask = (input_ids != 50256).unsqueeze(2).float()
        weights = mask / mask.sum(dim=1, keepdim=True)
        avg_pool = torch.sum(hidden_states * weights, 1)
        # avg_pool = torch.mean(hidden_states, 1)
        logits = self.linear(avg_pool)
        return logits


class CustomHead(nn.Module):
    def __init__(self, config, num_labels, dropout, start_layer=0):
        super().__init__()
        self.start_layer = start_layer
        self.num_layers = 12
        self.hidden_size = config.n_embd
        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(
            ((self.num_layers - self.start_layer + 1) // 2) *
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

    def forward(self, encoded_layers, mask):
        extracted_layers = torch.cat([
            torch.sum(encoded_layers[i] * mask.unsqueeze(2), dim=1)
            for i in range(self.start_layer, len(encoded_layers), 2)
        ], dim=1)
        out = self.dropout(self.pooler(extracted_layers))
        out = swish(out)
        return self.final_fc(out)


class GPT2ClassificationHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, cls_id, clf_dropout=0.4, n_class=8, head_start_layer=0):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.apply(self.init_weights)
        self.head = CustomHead(
            config, n_class, dropout=clf_dropout, start_layer=head_start_layer)
        self.cls_id = cls_id
        # self.head = AttentionHead(config, n_class)

    def forward(self, input_ids, past=None):
        hidden_states, _ = self.transformer(
            input_ids, None, None, past)
        # print(len(hidden_states),
        #       (input_ids == self.cls_id).float().sum())
        return self.head(hidden_states[1:], (input_ids == self.cls_id).float())

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        # Build new embeddings and initialize all new embeddings (in particular the special tokens)
        old_embed = self.transformer.wte
        self.transformer.wte = nn.Embedding(
            self.config.vocab_size + num_special_tokens, self.config.n_embd)
        self.transformer.wte.to(old_embed.weight.device)
        self.init_weights(self.transformer.wte)
        # Copy word embeddings from the previous weights
        self.transformer.wte.weight.data[:self.config.vocab_size, :] = (
            old_embed.weight.data[:self.config.vocab_size, :]
        )


def get_model(tokenizer):
    model = GPT2ClassificationHeadModel.from_pretrained(
        "gpt2", n_class=len(AUX_COLUMNS), clf_dropout=.1,
        cls_id=tokenizer.special_tokens["<cls>"],
        head_start_layer=7
    )
    print("CLS ID:", tokenizer.special_tokens["<cls>"])
    print("PAD ID:", tokenizer.special_tokens["<pad>"])
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    return model
