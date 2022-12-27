import torch
import numpy as np
import pandas as pd
from typing import List
from datasets import load_from_disk
from torch.utils.data import Dataset
from .utils import Preprocess_features
from transformers import AutoTokenizer


class RetrieverDataset(Dataset):
    """Dataset for the dense passage retrieval

    Attributes:
        self.mode (str): The type of the dataset to use
            This must be one of these: 'train', 'validation', or 'test'.
            This does not affect the mode of the model.
        self.tokenizer (): The tokenizer to tokenize questions and contexts from the imported dataset
        self.tokenized_passages (): In-batch negative samples
            If self.mode is 'train' or 'validation', self.construct_in_batch_negative_sampled_dataset() constructs this.
            Otherwise, this is None.
    """

    def __init__(self, config, mode="train"):
        assert mode in [
            "train",
            "validation",
            "test",
        ], f"RetrieverDataset > __init__: The mode {mode} does not exist. This must be one of these: 'train', 'validation', or 'test'."
        super(RetrieverDataset, self).__init__()
        self.config = config
        self.mode = mode

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])

        self.max_length = self.config["max_length"]
        self.stride = self.config["stride"]
        print(f"RetrieverDataset > __init__: The max length is set: {self.max_length}")
        print(f"RetrieverDataset > __init__: The stride is set: {self.stride}")

        self.PE = Preprocess_features(self.tokenizer, self.max_length, self.stride)

        if mode == "train":
            print(
                "RetrieverDataset > __init__: You are currently in the TRAINING process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["train"]
            (
                self.tokenized_passages,
                self.tokenized_questions,
            ) = self.construct_in_batch_negative_sampled_dataset()
        elif mode == "validation":
            print(
                "RetrieverDataset > __init__: You are currently in the VALIDATION process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["validation"]
            (
                self.tokenized_passages,
                self.tokenized_questions,
            ) = self.construct_in_batch_negative_sampled_dataset()
        elif mode == "test":
            print(
                "RetrieverDataset > __init__: You are currently in the TEST process. It will NOT construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["test_data_path"])["validation"]
            self.tokenized_questions = self.tokenizer(
                self.dataset["question"], padding=True, return_tensors="pt"
            )

    def __getitem__(self, index):
        if self.mode in ["train", "validation"]:
            return (
                self.tokenized_passages["input_ids"][index],
                self.tokenized_passages["attention_mask"][index],
                self.tokenized_passages["token_type_ids"][index],
                self.tokenized_questions["input_ids"][index],
                self.tokenized_questions["attention_mask"][index],
                self.tokenized_questions["token_type_ids"][index],
            )
        elif self.mode == "test":
            return (
                self.tokenized_questions["input_ids"][index],
                self.tokenized_questions["attention_mask"][index],
                self.tokenized_questions["token_type_ids"][index],
            )

    def __len__(self):
        return len(self.dataset)

    def construct_in_batch_negative_sampled_dataset(self):
        column_names = self.dataset.column_names
        tokenized_passages = self.dataset.map(
            self.PE.process,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=True,
        )
        questions = tokenized_passages["questions"]

        Passages = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        Questions = []

        for passage, question in zip(tokenized_passages, questions):
            if question != None:
                Passages["input_ids"].append(passage["input_ids"])
                Passages["attention_mask"].append(passage["attention_mask"])
                Passages["token_type_ids"].append(passage["token_type_ids"])
                Questions.append(question)

        tokenized_questions = self.tokenizer(
            Questions, padding=True, return_tensors="pt"
        )

        tokenized_passages = {k: torch.tensor(v) for k, v in Passages.items()}

        return tokenized_passages, tokenized_questions


class WikiDataset(Dataset):
    """Dataset for wikipedia documents

    Attributes:
        self.contexts (): Encoded wikipedia documents
    """

    def __init__(self, config, tokenizer):
        super(WikiDataset, self).__init__()
        self.corpus = pd.read_csv(config["corpus_path"])["text"]
        self.preprocess_corpus()

        self.contexts = tokenizer(
            self.corpus,
            max_length=config["max_length"],
            padding="max_length",
            truncation=True,
            stride=config["stride"],
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )  # (number of subdocuments, max length)

    def __len__(self):
        return len(self.contexts["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.contexts["input_ids"][idx],
            "attention_mask": self.contexts["attention_mask"][idx],
            "token_type_ids": self.contexts["token_type_ids"][idx],
        }

    def preprocess_corpus(self):
        """Preprocess the contexts in the Wikipedia corpus

        You can customize the following preprocessing approach.
        """
        self.corpus = [context.replace("\\n", "") for context in self.corpus]
