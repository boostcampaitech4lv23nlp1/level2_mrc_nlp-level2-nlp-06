import os
import yaml
import torch
import wandb
import argparse
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from utils import TOPK, set_seed
from model import DenseRetriever
from torch.utils.data import DataLoader
from transformers import TrainingArguments, get_linear_schedule_with_warmup
from dataset import RetrieverDataset, WikiDataset, HardNegatives


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

        if self.config["use_multiple_datasets"]:
            self.train_datasets = RetrieverDataset(self.config, mode="korquad")
        else:
            self.train_datasets = RetrieverDataset(self.config, mode="train")
        self.valid_datasets = RetrieverDataset(self.config, mode="validation")

        self.p_encoder = DenseRetriever(self.config).to(config["device"])
        if config["p_encoder_load_path"] and os.path.exists(config["p_encoder_load_path"]):
            print(
                f"retriever > train.py > main: Saved passage encoder file {config['p_encoder_load_path']} is found."
            )
            print("Load the pre-trained passage encoder...")
            self.p_encoder.load_state_dict(torch.load(config["p_encoder_load_path"]))
        if config["eval_topk"]:
            self.wikidataset = WikiDataset(config=config, tokenizer=self.train_datasets.tokenizer)
            self.wikiloader = DataLoader(self.wikidataset, batch_size=16, shuffle=False)
            self.topk = TOPK(config, self.train_datasets.tokenizer)
        
        if self.config["hard_negative_nums"] > 0:
            self.hn_dataset = HardNegatives(
                config=self.config, 
                tokenizer=self.train_datasets.tokenizer,
                max_length=config["max_length"],
                stride=config["stride"]
            )
            self.hn_dataset.construct_hard_negatives(
                dataset=self.train_datasets.dataset, 
                tokenized_passages=self.train_datasets.tokenized_passages
            )


    def train(self):
        train_dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"])
        valid_dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])
        hn_num = self.config["hard_negative_nums"]
        if hn_num > 0:
            hn_loader = DataLoader(
                dataset=self.hn_dataset, 
                batch_size=self.config["batch_size"]*self.config["hard_negative_nums"]
            )
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config["weight_decay"]},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
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
        torch.cuda.empty_cache()

        for epoch in tqdm(range(self.config["epochs"])):
            train_loss = 0
            valid_loss = 0
            if hn_num > 0:
                hn_iter = iter(hn_loader)
            for batch in tqdm(train_dataloader):
                batch_size = batch[0].shape[0]
                self.p_encoder.train()
                _, q_output, sim_scores = self.forward_step(batch)
                if hn_num > 0:
                    hn_batch = next(hn_iter)
                    hn_output = self.p_encoder(
                        hn_batch[0].to(self.args.device), 
                        hn_batch[1].to(self.args.device), 
                        hn_batch[2].to(self.args.device)
                    )
                    del hn_batch
                    temp = [hn_output[i*hn_num:(i+1)*hn_num].unsqueeze(0) for i in range(batch_size)]
                    hn_output = torch.cat(temp, dim=0)
                    temp = q_output.unsqueeze(1)
                    score = torch.bmm(hn_output, temp.transpose(1, 2)).squeeze(1)
                    
                    sim_scores = torch.cat((sim_scores, score), dim=1)
                
                targets = torch.arange(0, batch_size).long()
                targets = targets.to(self.args.device)

                sim_scores = F.log_softmax(sim_scores, dim=-1)

                loss = F.nll_loss(sim_scores, targets)
                train_loss += loss
                wandb.log({"train_loss": loss, "epoch": epoch})

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.p_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()
            wandb.log({"train_loss_per_epoch": train_loss / len(train_dataloader)})

            for batch in tqdm(valid_dataloader):
                self.p_encoder.eval()
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

            if config["eval_topk"]:
                print("\n*** CHECKING THE TRAIN & VALIDATION ACCURACY ***\n")
                p_outputs = self.topk.get_passage_outputs(self.p_encoder, epoch)
                scores, label_outputs = self.topk.get_results(self.p_encoder, train_dataloader, p_outputs)
                [train_top5, train_top20, train_top100], _ = self.topk.get_topk_results(p_outputs, scores, label_outputs, [5, 20, 100])
                scores, label_outputs = self.topk.get_results(self.p_encoder, valid_dataloader, p_outputs)
                [valid_top5, valid_top20, valid_top100], _ = self.topk.get_topk_results(p_outputs, scores, label_outputs, [5, 20, 100])
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
        torch.save(self.p_encoder.state_dict(), f"{self.config['p_encoder_save_path'][:-3]}-{epoch}-{valid_loss:.6f}.pt")



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
