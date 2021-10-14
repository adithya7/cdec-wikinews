"""
dataset preprocessing
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import random
import logging


def preprocess_eval_dataset(subtopics: List[List], coref_pairs: Dict, doc_dict: Dict, out_path: Path):
    """
    take raw dataset and create evaluation split
    """
    all_mention_pairs = []
    for subtopic_docs in subtopics:
        doc_pairs = []
        for i in range(len(subtopic_docs)):
            for j in range(i + 1, len(subtopic_docs)):
                doc_pairs += [(subtopic_docs[i], subtopic_docs[j])]
        for doc1, doc2 in doc_pairs:
            for mention1 in doc_dict[doc1]["mentions"]:
                for mention2 in doc_dict[doc2]["mentions"]:
                    m1_idx = (doc1, mention1["begin"], mention1["end"])
                    m2_idx = (doc2, mention2["begin"], mention2["end"])
                    if (m1_idx, m2_idx) in coref_pairs:
                        assert coref_pairs[(m1_idx, m2_idx)] == [mention1["sentence"], mention2["sentence"]]
                        all_mention_pairs += [
                            {
                                "doc1": doc1,
                                "doc2": doc2,
                                "sentence1": mention1["sentence"],
                                "sentence2": mention2["sentence"],
                                "label": "coreference",
                            }
                        ]
                    else:
                        all_mention_pairs += [
                            {
                                "doc1": doc1,
                                "doc2": doc2,
                                "sentence1": mention1["sentence"],
                                "sentence2": mention2["sentence"],
                                "label": "non-coreference",
                            }
                        ]

    with open(out_path, "w") as wf:
        json.dump(all_mention_pairs, wf, indent=2)


def collect_pos_neg_pairs(subtopics: List[List], coref_pairs: Dict, doc_dict: Dict):

    positive_pairs = []
    soft_candidate_negative_pairs = []
    hard_candidate_negative_pairs = []

    docs = []
    for subtopic_docs in subtopics:
        docs += subtopic_docs

    coref_sent_pairs = set()
    for mention_pair, sentence_pair in coref_pairs.items():
        m1, m2 = mention_pair
        s1, s2 = sentence_pair
        if m1[0] not in docs or m2[0] not in docs:
            continue
        s1 = s1.replace("<E> ", "").replace(" </E>", "")
        s2 = s2.replace("<E> ", "").replace(" </E>", "")
        coref_sent_pairs.update([(s1, s2), (s2, s1)])

    for subtopic_docs in subtopics:
        doc_pairs = []
        for i in range(len(subtopic_docs)):
            for j in range(i + 1, len(subtopic_docs)):
                doc_pairs += [(subtopic_docs[i], subtopic_docs[j])]
        for doc1, doc2 in doc_pairs:
            for mention1 in doc_dict[doc1]["mentions"]:
                for mention2 in doc_dict[doc2]["mentions"]:
                    m1_idx = (doc1, mention1["begin"], mention1["end"])
                    m2_idx = (doc2, mention2["begin"], mention2["end"])
                    if (m1_idx, m2_idx) in coref_pairs:
                        assert coref_pairs[(m1_idx, m2_idx)] == [mention1["sentence"], mention2["sentence"]]
                        positive_pairs += [
                            {
                                "doc1": doc1,
                                "doc2": doc2,
                                "sentence1": mention1["sentence"],
                                "sentence2": mention2["sentence"],
                                "label": "coreference",
                            },
                            {
                                "doc1": doc2,
                                "doc2": doc1,
                                "sentence1": mention2["sentence"],
                                "sentence2": mention1["sentence"],
                                "label": "coreference",
                            },
                        ]
                    else:
                        s1 = mention1["sentence"].replace("<E> ", "").replace(" </E>", "")
                        s2 = mention2["sentence"].replace("<E> ", "").replace(" </E>", "")
                        if (s1, s2) in coref_sent_pairs or (s2, s1) in coref_sent_pairs:
                            # coref link exists between a different mention pair between the same sentence
                            hard_candidate_negative_pairs += [
                                {
                                    "doc1": doc1,
                                    "doc2": doc2,
                                    "sentence1": mention1["sentence"],
                                    "sentence2": mention2["sentence"],
                                    "label": "non-coreference",
                                },
                                {
                                    "doc1": doc2,
                                    "doc2": doc1,
                                    "sentence1": mention2["sentence"],
                                    "sentence2": mention1["sentence"],
                                    "label": "non-coreference",
                                },
                            ]
                        else:
                            soft_candidate_negative_pairs += [
                                {
                                    "doc1": doc1,
                                    "doc2": doc2,
                                    "sentence1": mention1["sentence"],
                                    "sentence2": mention2["sentence"],
                                    "label": "non-coreference",
                                },
                                {
                                    "doc1": doc2,
                                    "doc2": doc1,
                                    "sentence1": mention2["sentence"],
                                    "sentence2": mention1["sentence"],
                                    "label": "non-coreference",
                                },
                            ]

    random.shuffle(hard_candidate_negative_pairs)
    random.shuffle(soft_candidate_negative_pairs)

    negative_pairs = hard_candidate_negative_pairs[: 5 * len(positive_pairs)]
    negative_pairs += soft_candidate_negative_pairs[: 5 * len(positive_pairs)]

    train_pairs = positive_pairs + negative_pairs

    return train_pairs


def preprocess_train_dataset(
    subtopics: List[List], coref_pairs: Dict, doc_dict: Dict, out_dir: Path, seed: int, k: int = 5
):

    random.seed(seed)

    # full train
    all_train_pairs = collect_pos_neg_pairs(subtopics, coref_pairs, doc_dict)

    with open(out_dir / f"train.json", "w") as wf:
        json.dump(all_train_pairs, wf, indent=2)

    # k-fold cross validation
    assert len(subtopics) % k == 0
    dev_size = len(subtopics) / k

    for i in range(k):
        cv_train_subtopics = []
        cv_dev_subtopics = []
        for j in range(len(subtopics)):
            if j >= i * dev_size and j < (i + 1) * dev_size:
                cv_dev_subtopics += [subtopics[j]]
            else:
                cv_train_subtopics += [subtopics[j]]

        logging.info(f"(cross-validation) train: {len(cv_train_subtopics)}")
        logging.info(f"(cross-validation) dev: {len(cv_dev_subtopics)}")

        cv_train_pairs = collect_pos_neg_pairs(cv_train_subtopics, coref_pairs, doc_dict)
        preprocess_eval_dataset(cv_dev_subtopics, coref_pairs, doc_dict, out_dir / f"dev_{i}.json")

        with open(out_dir / f"train_{i}.json", "w") as wf:
            json.dump(cv_train_pairs, wf, indent=2)

        with open(out_dir / f"train_{i}_subtopics.txt", "w") as wf:
            wf.write("\n".join([" ".join(st) for st in cv_train_subtopics]))
        with open(out_dir / f"dev_{i}_subtopics.txt", "w") as wf:
            wf.write("\n".join([" ".join(st) for st in cv_dev_subtopics]))


def load_subtopics(splits_path: Path, split: str):
    docs = []
    subtopics = []
    with open(splits_path / f"{split}_subtopics.txt", "r") as rf:
        for line in rf:
            subtopic_docs = line.strip().split()
            docs += subtopic_docs
            subtopics += [subtopic_docs]
    return docs, subtopics


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="preprocess data to create train, test splits")
    parser.add_argument("-coref_pairs", type=Path, help="path to all coref pairs json")
    parser.add_argument("-docs", type=Path, help="path to all docs dir")
    parser.add_argument("-splits", type=Path, help="path to train, test subtopics")
    parser.add_argument("-out_dir", type=Path, help="path to output directory")
    parser.add_argument("-seed", type=int, default=31, help="random seed")

    args = parser.parse_args()

    coref_pairs = {}
    with open(args.coref_pairs, "r") as rf:
        data = json.load(rf)
        for x in data:
            coref_pairs[(tuple(x["mention_spans"][0]), tuple(x["mention_spans"][1]))] = x["sentences"]

    doc_dict = {}
    for file_path in args.docs.iterdir():
        doc_name = file_path.stem
        with open(file_path, "r") as rf:
            doc_dict[doc_name] = json.load(rf)

    train_docs, train_subtopics = load_subtopics(args.splits, "train")
    test_docs, test_subtopics = load_subtopics(args.splits, "test")

    args.out_dir.mkdir(exist_ok=True)
    preprocess_train_dataset(train_subtopics, coref_pairs, doc_dict, args.out_dir, args.seed)
    preprocess_eval_dataset(test_subtopics, coref_pairs, doc_dict, args.out_dir / "test_pairs.json")
