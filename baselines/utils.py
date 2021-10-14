import json
from pathlib import Path
import logging
from typing import List, Tuple

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int]) -> None:
        super().__init__()

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        assert len(self.sentences1) == len(self.sentences2)
        self.labels = labels

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, index: int):
        return self.sentences1[index], self.sentences2[index], self.labels[index]


def load_dataset(train_path: Path, dev_path: Path = None):
    train_data, dev_data = [], []
    label2idx = {"non-coreference": 0, "coreference": 1}

    with open(train_path, "r") as rf:
        data = json.load(rf)
        sentences1, sentences2, labels = [], [], []
        for x in data:
            sentences1 += [x["sentence1"]]
            sentences2 += [x["sentence2"]]
            labels += [label2idx[x["label"]]]

        train_data = CustomDataset(sentences1, sentences2, labels)

    if dev_path is not None:
        with open(dev_path, "r") as rf:
            data = json.load(rf)
            sentences1, sentences2, labels = [], [], []
            for x in data:
                sentences1 += [x["sentence1"]]
                sentences2 += [x["sentence2"]]
                labels += [label2idx[x["label"]]]

            dev_data = CustomDataset(sentences1, sentences2, labels)

    return train_data, dev_data, label2idx


def load_eval_dataset(processed_dataset_path: Path):
    eval_data = []
    label2idx = {"non-coreference": 0, "coreference": 1}

    with open(processed_dataset_path, "r") as rf:
        data = json.load(rf)
        sentences1, sentences2, labels = [], [], []
        for x in data:
            sentences1 += [x["sentence1"]]
            sentences2 += [x["sentence2"]]
            labels += [label2idx[x["label"]]]

        eval_data = CustomDataset(sentences1, sentences2, labels)

    return eval_data, label2idx


def write_predictions(processed_dataset_path: Path, pred_labels: List[int], out_path: Path):
    label2idx = {"non-coreference": 0, "coreference": 1}
    idx2label = {0: "non-coreference", 1: "coreference"}

    sentences1, sentences2, labels = [], [], []
    with open(processed_dataset_path, "r") as rf:
        data = json.load(rf)
        for x in data:
            sentences1 += [x["sentence1"]]
            sentences2 += [x["sentence2"]]
            labels += [label2idx[x["label"]]]

    assert len(labels) == len(pred_labels)

    out_data = []
    for s1, s2, gold_label, pred_label in zip(sentences1, sentences2, labels, pred_labels):
        out_data += [
            {
                "sentence1": s1,
                "sentence2": s2,
                "gold_label": idx2label[gold_label],
                "pred_label": idx2label[pred_label],
            }
        ]

    with open(out_path, "w") as wf:
        json.dump(out_data, wf, indent=2)
