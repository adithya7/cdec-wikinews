from typing import List, Tuple

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer


class CorefClassifier(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        max_seq_length: int = 256,
        special_tag_tokens: List[str] = None,
    ) -> None:
        super().__init__()

        self.bert_model_name = bert_model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.special_tag_tokens = special_tag_tokens
        self.config = {
            "bert_model_name": self.bert_model_name,
            "num_labels": self.num_labels,
            "max_seq_length": self.max_seq_length,
            "special_tag_tokens": self.special_tag_tokens,
        }

        self.auto_model = AutoModel.from_pretrained(bert_model_name)
        self.auto_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(
            in_features=self.auto_model.config.hidden_size, out_features=self.num_labels
        )

        if self.special_tag_tokens is not None:
            special_tokens = {"additional_special_tokens": self.special_tag_tokens}
            self.auto_tokenizer.add_special_tokens(special_tokens)
            self.auto_model.resize_token_embeddings(len(self.auto_tokenizer))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def mean_pooling(self, model_output, attention_mask, input_ids):
        """
        Pool event tag embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        special_event_tag_id = self.auto_tokenizer.additional_special_tokens_ids[0]
        """ extract embedding of the special event tag """
        event_tag_mask_expanded = (
            (input_ids == special_event_tag_id).unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        special_event_token_embeddings = token_embeddings * event_tag_mask_expanded

        sum_embeddings = torch.sum(special_event_token_embeddings, 1)
        sum_mask = event_tag_mask_expanded.sum(1)

        # sum_embeddings = torch.sum(special_event_token_embeddings * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        return sentence_embeddings

    def forward(self, sentences1: List[str], sentences2: List[str], labels: List[int] = None):

        tokenized_sentences = self.auto_tokenizer(
            sentences1, sentences2, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        model_output = self.auto_model(**tokenized_sentences)
        sentence_embeddings = self.mean_pooling(
            model_output, tokenized_sentences["attention_mask"], tokenized_sentences["input_ids"]
        )
        logits = self.classifier(sentence_embeddings)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            target_labels = labels.to(self.device)
            loss = loss_fn(logits, target_labels)

        return logits, loss

    def save(self, save_path: str):
        model_state = {"state_dict": self.state_dict(), "config": self.config}
        torch.save(model_state, save_path)

    @classmethod
    def load(cls, model_path: str):
        model_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_args = model_state["config"]
        model = cls(**model_args)
        model.load_state_dict(model_state["state_dict"])
        return model
