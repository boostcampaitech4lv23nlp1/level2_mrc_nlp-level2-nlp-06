import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk


class RetrieverDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
        self.dataset = load_from_disk("../../../input/data/train_dataset/")
        self.negative_sampled_passage_batch = None
        self.tokenized_questions = None
        if train:
            print(
                "RetrieverDataset > __init__: You are currently in the TRAINING process. It will construct in-batch negative samples."
            )
            self.dataset = self.dataset["train"]
            self.construct_in_batch_negative_sampled_dataset()
        else:
            pass

    def __getitem__(self, index):
        if self.train:
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
        if self.train:
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
        else:
            pass
