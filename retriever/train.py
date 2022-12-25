from dataset import RetrieverDataset
from model import DenseRetriever

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torchmetrics import Accuracy
from transformers import TrainingArguments, get_linear_schedule_with_warmup, AutoTokenizer
import random
import numpy as np
import argparse
import yaml
import wandb
from tqdm import tqdm
from datasets import load_from_disk


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


class RetrieverTrainer:
    def __init__(self, config):
        self.config = config
        
        self.args = TrainingArguments(
            output_dir=self.config["output_dir"],
            evaluation_strategy="epoch",
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            weight_decay=self.config["weight_decay"],
            report_to=["wandb"]
        )

        self.train_datasets = RetrieverDataset(self.config)
        self.valid_datasets = RetrieverDataset(self.config, mode="validation")

        self.p_encoder = DenseRetriever(self.config).to(config["device"])
        self.q_encoder = DenseRetriever(self.config).to(config["device"])

        self.num_neg = self.config["num_negative_passages_per_question"]

        tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
        self.tokenized_corpus = []
        for context in tqdm(pd.read_csv(config["corpus_path"])["text"]):
            tokenized_context = tokenizer(
                context,
                truncation=True,
                stride=config["stride"],
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.tokenized_corpus.extend([c for c in tokenized_context])


    def train(self):
        train_dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"], shuffle=True)
        valid_dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["learning_rate"],
            eps=self.args.adam_epsilon
        )
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config["warmup_ratio"]*t_total),
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for epoch in tqdm(range(self.config["epochs"])):
            for batch in tqdm(train_dataloader):
                self.p_encoder.train()
                self.q_encoder.train()
                _, _, sim_scores = self.forward_step(batch)
                targets = torch.arange(0, batch[0].shape[0]).long()
                targets = targets.to(self.args.device)

                sim_scores = F.log_softmax(sim_scores, dim=-1)

                # accuracy = Accuracy(task="multiclass", num_classes=batch[0].shape[0], top_k=1).to(config["device"])
                # acc = accuracy(sim_scores, targets)
                loss = F.nll_loss(sim_scores, targets)
                wandb.log({"train_loss": loss, "epoch": epoch})

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.q_encoder.zero_grad()
                self.p_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()
            print("\n*** CHECKING THE TRAINING ACCURACY ***\n")
            train_accuracy = self.count_match(mode="train")
            print("*** TRAIN ACCURACY:", train_accuracy)
            wandb.log({"train_accuracy": train_accuracy, "full_train_loss": loss, "full_epoch": epoch})

            for batch in tqdm(valid_dataloader):
                self.p_encoder.eval()
                self.q_encoder.eval()
                with torch.no_grad():
                    _, _, sim_scores = self.forward_step(batch)
                targets = torch.arange(0, batch[0].shape[0]).long()
                targets = targets.to(self.args.device)
                
                sim_scores = F.log_softmax(sim_scores, dim=-1)
                targets = torch.zeros(batch[0].shape[0]).long()
                targets = targets.to(self.args.device)

                # accuracy = Accuracy(task="multiclass", num_classes=batch[0].shape[0], top_k=1).to(config["device"])
                # acc = accuracy(sim_scores, targets)
                loss = F.nll_loss(sim_scores, targets)

                wandb.log({"valid_loss": loss, "epoch": epoch})
                
                torch.cuda.empty_cache()
            print("\n*** CHECKING THE VALIDATION ACCURACY ***\n")
            valid_accuracy = self.count_match(mode="validation")
            print("*** VALIDATION ACCURACY:", valid_accuracy)
            wandb.log({"valid_accuracy": valid_accuracy, "full_valid_loss": loss, "full_epoch": epoch})


    def forward_step(self, batch):
        batch_size = batch[0].shape[0]

        p_inputs = {
            "input_ids": batch[0].view(batch_size, -1).to(self.args.device),
            "attention_mask": batch[1].view(batch_size, -1).to(self.args.device),
            "token_type_ids": batch[2].view(batch_size, -1).to(self.args.device)
        }

        q_inputs = {
            "input_ids": batch[3].to(self.args.device),
            "attention_mask": batch[4].to(self.args.device),
            "token_type_ids": batch[5].to(self.args.device)
        }

        del batch
        torch.cuda.empty_cache()
        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        sim_scores = torch.matmul(q_outputs, p_outputs.T).squeeze()
        sim_scores = sim_scores.view(batch_size, -1)

        del q_inputs, p_inputs

        return p_outputs, q_outputs, sim_scores


    def count_match(self, mode):
        number_of_matches = 0

        self.p_encoder.eval()
        self.q_encoder.eval()

        context_embeddings = []

        for tokenized_context in tqdm(self.tokenized_corpus):
            with torch.no_grad():
                tokenized_context = {k: v.to(self.args.device) for k, v in tokenized_context.items()}
                context_embedding = self.p_encoder(**tokenized_context).detach().cpu()
                del tokenized_context
                context_embeddings.append(context_embedding)
                del context_embedding
        context_embeddings = torch.stack(context_embeddings)
        context_embeddings = context_embeddings.squeeze() # (number of contexts, max length)
        context_embeddings = context_embeddings.to(self.args.device)

        if mode == "train":
            dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"])
        elif mode == "validation":
            dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])

        for batch in tqdm(dataloader):
            question_embeddings = []
            with torch.no_grad():
                p_outputs, q_outputs, _ = self.forward_step(batch)
            gold_passage_embeddings = p_outputs[:, 0]
            question_embeddings = q_outputs.squeeze()
            del p_outputs
            del q_outputs

            # Matrix multiplication between (batch size, max length) and (max length, number of contexts) -> (batch size, number of contexts)
            sim_scores = torch.mm(question_embeddings, torch.transpose(context_embeddings, 0, 1))
            del question_embeddings
            for i, pair in enumerate(sim_scores):
                top_k_context_embeddings = context_embeddings[torch.argsort(pair, dim=-1, descending=True)[:self.config["top_k"]]]
                if gold_passage_embeddings[i] in top_k_context_embeddings:
                    number_of_matches += 1
            torch.cuda.empty_cache()
        print("*** Length", len(dataloader)* self.config["batch_size"])
        return number_of_matches / (len(dataloader) * self.config["batch_size"])


    def save_models(self):
        torch.save(self.p_encoder.state_dict(), self.config["p_encoder_save_path"])
        torch.save(self.q_encoder.state_dict(), self.config["q_encoder_save_path"])


def main(config):
    set_seed(config["random_seed"])

    wandb.init(
            project=config["wandb_project"], 
            name=config["wandb_name"], 
            notes=config["wandb_note"], 
            entity=config["wandb_entity"], 
            group=config["wandb_group"],
            config=config
        )

    trainer = RetrieverTrainer(config)
    trainer.train()
    trainer.save_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
