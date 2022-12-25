import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import Dataset
from .utils import Preprocess_features


class RetrieverDataset(Dataset):
    """Dataset for the dense passage retrieval

    CAUTION! The development for testing and inferencing is still in progress.

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

        self.max_length = 512
        self.stride = 128
        self.PE = Preprocess_features(self.tokenizer, self.max_length, self.stride)
        
        if mode == "train":
            print(
                "RetrieverDataset > __init__: You are currently in the TRAINING process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["train"]
        elif mode == "validation":
            print(
                "RetrieverDataset > __init__: You are currently in the VALIDATION process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["validation"]
        elif mode == "test":
            pass
        self.tokenized_passages, self.tokenized_questions = self.construct_in_batch_negative_sampled_dataset()

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
        else:
            pass

    def __len__(self):
        return len(self.tokenized_passages)

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
            Questions,
            padding=True,
            return_tensors="pt"
        )
        
        tokenized_passages = {k: torch.tensor(v) for k, v in Passages.items()}
        
        return tokenized_passages, tokenized_questions
        
