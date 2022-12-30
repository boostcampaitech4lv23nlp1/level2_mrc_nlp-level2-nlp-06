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
from dataset import RetrieverDataset, WikiDataset, AugmentedRetrieverDataset
        

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
            self.train_datasets = AugmentedRetrieverDataset(self.config)
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

        self.wikidataset = WikiDataset(config=config, tokenizer=self.train_datasets.tokenizer)
        self.wikiloader = DataLoader(self.wikidataset, batch_size=16, shuffle=False)
        
        self.topk = TOPK(config, self.train_datasets.tokenizer)


    def train(self):
        train_dataloader = DataLoader(self.train_datasets, batch_size=self.config["batch_size"], shuffle=True)
        valid_dataloader = DataLoader(self.valid_datasets, batch_size=self.config["batch_size"])
        
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
            for batch in tqdm(train_dataloader):
                self.p_encoder.train()
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

            print("\n*** CHECKING THE TRAIN & VALIDATION ACCURACY ***\n")
            train_top5, train_top20, train_top100 = self.topk.get_results(self.p_encoder, epoch, train_dataloader, True)
            valid_top5, valid_top20, valid_top100 = self.topk.get_results(self.p_encoder, epoch, valid_dataloader, False)
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
