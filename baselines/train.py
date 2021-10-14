import argparse
import json
from pathlib import Path
import logging
import math
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers

from utils import load_dataset, load_eval_dataset, write_predictions
from models import CorefClassifier

logger = logging.getLogger(__name__)


def evaluator(model, data, return_preds: bool = False):
    model.eval()

    pred_labels, gold_labels = [], []
    for _, batch in tqdm(enumerate(data)):
        batch_sentences1, batch_sentences2, batch_labels = batch
        logits, _ = model(batch_sentences1, batch_sentences2)
        logits = nn.functional.softmax(logits, dim=1)
        pred_labels += list(np.argmax(logits.detach().cpu().numpy(), axis=1))
        gold_labels += list(batch_labels.numpy())

    pred_labels = np.array(pred_labels)
    gold_labels = np.array(gold_labels)

    acc = np.sum(pred_labels == gold_labels) / len(gold_labels)
    logger.info("Accuracy: {:.2f}".format(acc * 100))

    tp = np.sum(np.take(gold_labels, np.nonzero(pred_labels)) == 1)
    fp = np.sum(np.take(gold_labels, np.nonzero(pred_labels)) == 0)
    tn = np.sum(np.take(gold_labels, np.nonzero(pred_labels == 0)) == 0)
    fn = np.sum(np.take(gold_labels, np.nonzero(pred_labels == 0)) == 1)

    coref_precision = tp / (tp + fp)
    coref_recall = tp / (tp + fn)
    coref_f1 = (2 * coref_precision * coref_recall) / (coref_precision + coref_recall)
    logger.info("(Coref) Precision: {:.2f}".format(coref_precision * 100))
    logger.info("(Coref) Recall: {:.2f}".format(coref_recall * 100))
    logger.info("(Coref) F1: {:.2f}".format(coref_f1 * 100))

    non_coref_precision = tn / (tn + fn)
    non_coref_recall = tn / (tn + fp)
    non_coref_f1 = (2 * non_coref_precision * non_coref_recall) / (non_coref_precision + non_coref_recall)
    logger.info("(Non-Coref) Precision: {:.2f}".format(non_coref_precision * 100))
    logger.info("(Non-Coref) Recall: {:.2f}".format(non_coref_recall * 100))
    logger.info("(Non-Coref) F1: {:.2f}".format(non_coref_f1 * 100))

    if return_preds:
        return acc, coref_precision, coref_recall, coref_f1, pred_labels

    return acc, coref_precision, coref_recall, coref_f1


def train(args):

    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir.mkdir(exist_ok=True)
    model_save_path = args.save_dir / f"coref_classifier_{time_stamp}.bin"
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/coref_train_%s.log" % time_stamp),],
    )

    expt_config = json.load(open(args.config, "r"))
    train_data, dev_data, label2idx = load_dataset(args.train_path, args.dev_path)

    run_eval = False
    if len(dev_data) > 0:
        run_eval = True

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    if run_eval:
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)

    model = CorefClassifier(
        bert_model_name=args.bert_model,
        num_labels=len(label2idx),
        max_seq_length=args.max_seq,
        special_tag_tokens=expt_config["special_tags"],
    )
    if torch.cuda.is_available():
        model = model.cuda()

    model_parameters = model.named_parameters()
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    num_train_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)  # 10% of train data for warm-up
    logger.info(f"Warmup-steps: {warmup_steps}")

    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )

    steps_per_epoch = len(train_dataloader)
    best_f1 = 0.0

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch}")

        epoch_loss = 0.0
        for step_idx, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            batch_sentences1, batch_sentences2, batch_labels = batch
            _, loss = model(batch_sentences1, batch_sentences2, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if run_eval:
                if (step_idx + 1) % args.eval_steps == 0:
                    logging.info(f"evaluating at epoch {epoch}, step: {step_idx}")
                    acc, precision, recall, f1 = evaluator(model, dev_dataloader)
                    if f1 > best_f1:
                        best_f1 = f1
                        logger.info("Saving the model to %s" % model_save_path)
                        model.save(model_save_path)

        logger.info(f"Epoch {epoch} loss: {epoch_loss}")
        if run_eval:
            acc, precision, recall, f1 = evaluator(model, dev_dataloader)
            if f1 > best_f1:
                best_f1 = f1
                logger.info("Saving the model to %s" % model_save_path)
                model.save(model_save_path)
        else:
            logger.info("Saving the model to %s" % model_save_path)
            model.save(model_save_path)


def inference(args):
    """
    load pretrained model and run inference
    """
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/coref_inference_%s.log" % time_stamp),],
    )

    eval_data, label2idx = load_eval_dataset(args.data_path)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size)

    model = CorefClassifier.load(args.model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.preds is None:
        acc, precision, recall, f1 = evaluator(model, eval_dataloader)
    else:
        acc, precision, recall, f1, pred_labels = evaluator(model, eval_dataloader, return_preds=True)
        write_predictions(args.data_path, pred_labels, args.preds)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train/eval pairwise coreference classifier")
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="train classifer")
    train_parser.add_argument("-train_path", type=Path, help="path to preprocessed train json")
    train_parser.add_argument("-dev_path", type=Path, default=None, help="path to preprocessed dev json")
    train_parser.add_argument("-save_dir", type=Path, help="path to save model")
    train_parser.add_argument("-bert_model", type=str, default="bert-base-uncased")
    train_parser.add_argument("-batch_size", type=int, default=16, help="train batch size")
    train_parser.add_argument("-epochs", type=int, default=5, help="number of train epochs")
    train_parser.add_argument("-eval_steps", type=int, default=300, help="evaluation steps")
    train_parser.add_argument("-max_seq", type=int, default=256, help="maximum sequence length")
    train_parser.add_argument("-lr", type=float, default=2e-5, help="learning rate")
    train_parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay")
    train_parser.add_argument("-config", type=Path, help="path to config")
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("inference", help="eval classifier")
    eval_parser.add_argument("-model_path", type=str, help="path to load model")
    eval_parser.add_argument("-data_path", type=str, help="path to eval json")
    eval_parser.add_argument("-batch_size", type=int, default=16, help="eval batch size")
    eval_parser.add_argument("-preds", type=Path, default=None, help="path to write model predictions")
    eval_parser.set_defaults(func=inference)

    args = parser.parse_args()
    args.func(args)
