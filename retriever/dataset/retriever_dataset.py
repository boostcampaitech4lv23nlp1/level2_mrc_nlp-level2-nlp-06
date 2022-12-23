import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk
from .utils import Preprocess_features


class RetrieverDataset(Dataset):
    """Dataset for the dense passage retrieval

    CAUTION! The development for testing and inferencing is still in progress.

    Attributes:
        self.mode (str): The type of the dataset to use
            This must be one of these: 'train', 'validation', or 'test'.
            This does not affect the mode of the model.
        self.tokenizer (): The tokenizer to tokenize questions and contexts from the imported dataset
        self.negative_sampled_passage_batch (): In-batch negative samples
            If self.mode is 'train' or 'validation', self.construct_in_batch_negative_sampled_dataset() constructs this.
            Otherwise, this is None.
    """
    def __init__(self, config, mode="train"):
        assert mode in [
            "train",
            "validation",
            "test",
        ], f"RetrieverDataset > __init__: The mode {mode} does not exist. This must be one of these: 'train', 'validation', or 'test'."
        self.config = config

        self.mode = mode

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
        print(
            f"RetrieverDataset > __init__: Number of negative passages per question is set: {config['num_negative_passages_per_question']}"
        )
        self.negative_sampled_passage_batch = None
        self.tokenized_questions = None
        self.max_length = 512
        self.stride = 128
        self.PE = Preprocess_features(self.tokenizer, self.max_length, self.stride)
        
        if mode == "train":
            print(
                "RetrieverDataset > __init__: You are currently in the TRAINING process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["train"]
            self.construct_in_batch_negative_sampled_dataset()
        elif mode == "validation":
            print(
                "RetrieverDataset > __init__: You are currently in the VALIDATION process. It will construct in-batch negative samples."
            )
            self.dataset = load_from_disk(config["train_data_path"])["validation"]
            self.construct_in_batch_negative_sampled_dataset()
        elif mode == "test":
            pass

    def __getitem__(self, index):
        if self.mode in ["train", "validation"]:
            return (
                self.negative_sampled_passage_batch["input_ids"][index],
                self.negative_sampled_passage_batch["attention_mask"][index],
                self.negative_sampled_passage_batch["token_type_ids"][index],
                self.tokenized_questions["input_ids"][index],
                self.tokenized_questions["attention_mask"][index],
                self.tokenized_questions["token_type_ids"][index],
            )
        else:
            pass

    def __len__(self):
        return len(self.dataset)

    def construct_negative_sampled_batch(self):
        corpus = np.array(list(set([example for example in self.dataset["context"]])))
        passage_batch = []

        for passage in self.dataset["context"]:
            while True:
                negative_passage_indices = np.random.randint(
                    len(corpus), size=self.config["num_negative_passages_per_question"]
                )

                if not passage in corpus[negative_passage_indices]:
                    negative_passage = corpus[negative_passage_indices]

                    passage_batch.append(passage)
                    passage_batch.extend(negative_passage)
                    break

        return passage_batch

    def construct_in_batch_negative_sampled_dataset(self):
        passage_batch = self.construct_negative_sampled_batch()

        tokenized_passage_batch = self.tokenizer(
            passage_batch,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_questions = self.tokenizer(
            self.dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        max_len = tokenized_passage_batch["input_ids"].size(-1)
        tokenized_passage_batch["input_ids"] = tokenized_passage_batch[
            "input_ids"
        ].view(-1, self.config["num_negative_passages_per_question"] + 1, max_len)
        tokenized_passage_batch["attention_mask"] = tokenized_passage_batch[
            "attention_mask"
        ].view(-1, self.config["num_negative_passages_per_question"] + 1, max_len)
        tokenized_passage_batch["token_type_ids"] = tokenized_passage_batch[
            "token_type_ids"
        ].view(-1, self.config["num_negative_passages_per_question"] + 1, max_len)

        self.negative_sampled_passage_batch = tokenized_passage_batch
        self.tokenized_questions = tokenized_questions
