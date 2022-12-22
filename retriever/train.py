from dataset import RetrieverDataset
from model import DenseRetriever

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import TrainingArguments, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import trange, tqdm
import argparse
import yaml


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    
class RetrieverTrainer:
    # Train 옵션 설정.
    def __init__(self, config):
        # yaml 이용해서 config 받아온다고 가정.
        self.config = config
        
        self.args = TrainingArguments(
            output_dir=self.config["output_dir"],
            evaluation_strategy="epoch",
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            weight_decay=self.config["weight_decay"]
        )
        
        self.train_datasets = RetrieverDataset(self.config)
        
        self.p_encoder = DenseRetriever(self.config).to(config["device"])
        self.q_encoder = DenseRetriever(self.config).to(config["device"])
        
        self.num_neg = self.config["num_negative_passages_per_question"]
        
        
    def train(self):
        train_dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"])
        # valid_dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])
        
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
        
        train_iterator = trange(int(self.config["epochs"]), desc="Epoch")
        for _ in train_iterator:
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()
                    _, _, sim_scores = self.forward_step(batch)
                    # In-batch negative 적용 시 바꿔야 하는 부분.
                    targets = torch.zeros(batch[0].shape[0]).long()
                    targets = targets.to(self.args.device)
                    
                    sim_scores = F.log_softmax(sim_scores, dim=-1)
                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")
                    
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()
                    
                    global_step += 1
                    
                    torch.cuda.empty_cache()
                    
            
            # with tqdm(valid_dataloader, unit="batch") as tepoch:
            #     for batch in tepoch:
            #         self.p_encoder.eval()
            #         self.q_encoder.eval()
            #         with torch.no_grad():
            #             _, _, sim_scores = self.forward_step(batch)
            #         sim_scores = F.log_softmax(sim_scores, dim=-1)
            #         loss = F.nll_loss(sim_scores, targets)
            #         tepoch.set_postfix(loss=f"{str(loss.item())}")
                    
            #         torch.cuda.empty_cache()

                    
    def forward_step(self, batch):
        batch_size = batch[0].shape[0]

        p_inputs = {
            "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(self.args.device),
            "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(self.args.device),
            "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(self.args.device)
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
        
        p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
        q_outputs = q_outputs.view(batch_size, 1, -1)
        
        sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
        sim_scores = sim_scores.view(batch_size, -1)
        
        del q_inputs, p_inputs
        
        return p_outputs, q_outputs, sim_scores
    
    def save_models(self):
        torch.save(self.p_encoder.state_dict(), self.config["p_encoder_save_path"])
        torch.save(self.q_encoder.state_dict(), self.config["q_encoder_save_path"])
    

def main(config):
    set_seed(config["random_seed"])
    
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
    