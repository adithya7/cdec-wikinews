"""
lemma-delta baseline from Upadhyay et al. 2016
"""
import argparse
import spacy
import json
import itertools
from pathlib import Path
import re
import numpy as np
import logging
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def load_docs(docs_dir: Path, subtopics_path: Path):

    docs = set()
    subtopics = []
    with open(subtopics_path, "r") as rf:
        for line in rf:
            subtopic_docs = line.strip().split()
            docs.update(subtopic_docs)
            subtopics += [subtopic_docs]

    logger.info(f"loaded {len(docs)} documents")
    logger.info(f"loaded {len(subtopics)} subtopics")

    doc2txt = {}
    for file_path in docs_dir.iterdir():
        doc_name = file_path.stem
        with open(file_path, "r") as rf:
            if doc_name in docs:
                doc2txt[doc_name] = json.load(rf)["text"]

    return doc2txt


def load_mention_pairs(json_path: Path):
    with open(json_path, "r") as rf:
        eval_data = json.load(rf)

    logger.info(f"{len(eval_data)} mention pairs")

    return eval_data


def run_lemma_baseline(doc2txt, eval_data, threshold):

    label2idx = {"non-coreference": 0, "coreference": 1}

    spacy_en_nlp = spacy.load("en_core_web_lg")

    doc_texts = []
    doc_names = []

    for doc, txt in doc2txt.items():
        doc_texts += [txt]
        doc_names += [doc]

    logging.info("computing document pair similarities")
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(doc_texts)

    doc_pair_scores = {}
    for doc1, doc2 in itertools.combinations_with_replacement(doc_names, 2):
        doc1_vectors = vectorizer.transform(iter([doc2txt[doc1]]))
        doc2_vectors = vectorizer.transform(iter([doc2txt[doc2]]))
        sim_score = cosine_similarity(doc1_vectors, doc2_vectors)
        doc_pair_scores[(doc1, doc2)] = sim_score
        doc_pair_scores[(doc2, doc1)] = sim_score

    logging.info("predicting labels")
    pred_labels, gold_labels = [], []
    for eval_sample in tqdm(eval_data):
        mention1 = re.search(r"<E> (.*) </E>", eval_sample["sentence1"]).group(1).lower()
        mention2 = re.search(r"<E> (.*) </E>", eval_sample["sentence2"]).group(1).lower()
        mention1_head_lemma = list(spacy_en_nlp(mention1).sents)[0].root.lemma
        mention2_head_lemma = list(spacy_en_nlp(mention2).sents)[0].root.lemma
        doc1 = eval_sample["doc1"]
        doc2 = eval_sample["doc2"]
        if (mention1_head_lemma == mention2_head_lemma) and (doc_pair_scores[(doc1, doc2)] >= threshold):
            pred_labels += [label2idx["coreference"]]
        else:
            pred_labels += [label2idx["non-coreference"]]
        gold_labels += [label2idx[eval_sample["label"]]]

    acc = np.sum(pred_labels == gold_labels) / len(gold_labels)
    precision = np.mean(np.take(gold_labels, np.nonzero(pred_labels)))
    recall = np.mean(np.take(pred_labels, np.nonzero(gold_labels)))
    f1 = (2 * precision * recall) / (precision + recall)

    logger.info("Accuracy: {:.2f}".format(acc * 100))
    logger.info("(Coref) Precision: {:.2f}".format(precision * 100))
    logger.info("(Coref) Recall: {:.2f}".format(recall * 100))
    logger.info("(Coref) F1: {:.2f}".format(f1 * 100))


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="lemma baseline")
    parser.add_argument("-docs", type=Path, help="documents directory")
    parser.add_argument("-subtopics", type=Path, help="path to subtopics list")
    parser.add_argument("-dev_path", type=Path, help="path to dev json")
    parser.add_argument(
        "-sim_threshold", type=float, default=0.0, help="tf-idf document similarity threshold"
    )

    args = parser.parse_args()

    doc2txt = load_docs(args.docs, args.subtopics)
    eval_data = load_mention_pairs(args.dev_path)

    run_lemma_baseline(doc2txt, eval_data, threshold=args.sim_threshold)
