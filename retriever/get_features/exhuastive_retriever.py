import sys
sys.path.append("../")
from model import DenseRetriever

import torch
import numpy as np
import argparse
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def main(config):
    p_encoder = DenseRetriever(config)
    q_encoder = DenseRetriever(config)
    
    p_encoder.eval()
    q_encoder.eval()
    
    p_encoder.to(config["device"])
    q_encoder.to(config["device"])
    
    dataset = load_from_disk("../input/data/train_dataset/")
    
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    
    corpus = [context for context in dataset["train"]["context"]]
    c_embs = []
    for c in tqdm(corpus):
        c = tokenizer(c, padding="max_length", truncation=True, return_tensors="pt")
        c = c.to(config["device"])
        with torch.no_grad():
            c_emb = p_encoder(**c).to("cpu").numpy()
            c_embs.append(c_emb)
    c_embs = torch.tensor(c_embs).squeeze()
    
    np.save(config["embedding_path"], c_embs.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    