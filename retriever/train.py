import os
import yaml
import torch
import wandb
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from model import DenseRetriever
from torch.utils.data import DataLoader
from dataset import RetrieverDataset, WikiDataset
from transformers import TrainingArguments, get_linear_schedule_with_warmup, AutoTokenizer


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
class TOPK:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.wiki_dataset = WikiDataset(config, tokenizer)
        self.wiki_dataloader = DataLoader(self.wiki_dataset, batch_size=config["batch_size"])
        
    def get_results(self, p_encoder, epoch, dataloader):
        p_encoder.eval()
        p_outputs = []
        for data in tqdm(self.wiki_dataloader):
            data = {k: v.to('cuda') for k, v in data.items()}
            with torch.no_grad():
                p_output = p_encoder(**data)
            p_outputs.append(p_output.cpu())
        p_outputs = torch.cat(p_outputs, dim=0)
        # Save corpus features.
        corpus_feature_paths = config["corpus_feature_path"].split(".")[0]
        corpus_feature_paths = corpus_feature_paths + str(epoch) + ".pickle"
        with open(corpus_feature_paths, "wb") as f:
            pickle.dump(p_outputs, f)
            
        q_outputs = []
        label_outputs = []
        for data in dataloader:
            with torch.no_grad():
                data = [d.to(config["device"]) for d in data]
                label_output = p_encoder(data[0], data[1], data[2])
                question_output = p_encoder(data[3], data[4], data[5])
            q_outputs.append(question_output.cpu())
            label_outputs.append(label_output.cpu())
        q_outputs = torch.cat(q_outputs, dim=0)
        label_outputs = torch.cat(label_outputs, dim=0)
        
        scores = torch.matmul(q_outputs, p_outputs.T)
        topk_indices = []
        for score in scores:
            topk_res = torch.topk(score, 100)
            topk_indices.append(topk_res.indices)
        top5 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 5)
        top20 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 20)
        top100 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 100)
        
        return top5, top20, top100
        
    def calc_wiki_accuracy(self, pred_context, label_context, indexes, k):
        correct = 0
        for i, index in enumerate(indexes):
            label = label_context[i]
            for idx in index[:k]:  # top-k
                if pred_context[idx].tolist() == label.tolist():
                    correct += 1

        return correct / len(indexes)
        


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

        self.train_datasets = RetrieverDataset(self.config, mode="train")
        self.valid_datasets = RetrieverDataset(self.config, mode="validation")

        self.p_encoder = DenseRetriever(self.config).to(config["device"])
        # self.q_encoder = DenseRetriever(self.config).to(config["device"])
        
        self.topk = TOPK(config, self.train_datasets.tokenizer)


    def train(self):
        train_dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"], shuffle=True)
        valid_dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            # {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            # {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
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
        # self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for epoch in tqdm(range(self.config["epochs"])):
            train_loss = 0
            valid_loss = 0
            for batch in tqdm(train_dataloader):
                self.p_encoder.train()
                # self.q_encoder.train()
                _, _, sim_scores = self.forward_step(batch)
                targets = torch.arange(0, batch[0].shape[0]).long()
                targets = targets.to(self.args.device)

                sim_scores = F.log_softmax(sim_scores, dim=-1)

                loss = F.nll_loss(sim_scores, targets)
                train_loss += loss
                wandb.log({"train_loss": loss, "epoch": epoch})

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # self.q_encoder.zero_grad()
                self.p_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()
            wandb.log({"train_loss_per_epoch": train_loss / len(train_dataloader)})

            for batch in tqdm(valid_dataloader):
                self.p_encoder.eval()
                # self.q_encoder.eval()
                with torch.no_grad():
                    _, _, sim_scores = self.forward_step(batch)
                targets = torch.arange(0, batch[0].shape[0]).long()
                targets = targets.to(self.args.device)
                
                sim_scores = F.log_softmax(sim_scores, dim=-1)

                loss = F.nll_loss(sim_scores, targets)
                valid_loss += loss
                
                torch.cuda.empty_cache()
            valid_loss /= len(valid_dataloader)
            wandb.log({"valid_loss_per_epoch": valid_loss})

            print("\n*** CHECKING THE TRAIN & VALIDATION ACCURACY ***\n")
            train_top5, train_top20, train_top100 = self.topk.get_results(self, self.p_encoder, epoch, train_dataloader)
            valid_top5, valid_top20, valid_top100 = self.topk.get_results(self, self.p_encoder, epoch, valid_dataloader)
            train_top5, train_top20, train_top100, valid_top5, valid_top20, valid_top100 = self.count_match()
            wandb.log({
                "train_top5 accuracy" : train_top5,
                "train_top20 accuracy" : train_top20,
                "train_top100 accuracy" : train_top100,
                "valid_top5 accuracy" : valid_top5,
                "valid_top20 accuracy" : valid_top20,
                "valid_top100 accuracy" : valid_top100,
            })

            print("\n*** SAVING THE CHECKPOINT ***\n")
            self.save_checkpoint(epoch, valid_loss)


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
        q_outputs = self.p_encoder(**q_inputs)

        sim_scores = torch.matmul(q_outputs, p_outputs.T).squeeze()
        sim_scores = sim_scores.view(batch_size, -1)

        del q_inputs, p_inputs

        return p_outputs, q_outputs, sim_scores


    def save_checkpoint(self, epoch, valid_loss):
        ## TODO: save_path로 디렉토리를 받도록 수정하기. 이를 위해선 inference 코드들이 수정되어야 함. -> inference용 config 만들기.
        torch.save(self.p_encoder.state_dict(), f"{self.config['p_encoder_save_path'][:-3]}-{epoch}-{valid_loss:.6f}.pt")
        # torch.save(self.q_encoder.state_dict(), f"{self.config['q_encoder_save_path'][:-3]}-{epoch}-{valid_loss:.6f}.pt")



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
