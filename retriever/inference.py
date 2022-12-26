import pandas as pd
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from model import DenseRetriever
import torch
from dataset import RetrieverDataset
from tqdm import tqdm
import pickle
import os
import argparse


class wiki_dataset(Dataset):
    def __init__(self, config, wiki_corpus, tokenizer):
        super(wiki_dataset, self).__init__()
        self.contexts = tokenizer(
            wiki_corpus,
            truncation=True,
            max_length=384,
            stride=config["stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
    def __len__(self):
        return len(self.contexts['input_ids'])
    
    def __getitem__(self, idx):
        return {"input_ids": self.contexts['input_ids'][idx],
                "attention_mask": self.contexts["attention_mask"][idx],
                "token_type_ids": self.contexts["token_type_ids"][idx]}

def main(config):
    # prepare wiki corpus
    df = pd.read_csv(config["corpus_path"])
    wiki_corpus = df["text"]
    wiki_corpus = [corpus.replace("\\n", "") for corpus in wiki_corpus]
    
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    print("process dataset...")
    wikidataset = wiki_dataset(config, wiki_corpus, tokenizer)
    print("done!")
    dataloader = DataLoader(wikidataset, batch_size=16)

    p_encoder = DenseRetriever(config)
    q_encoder = DenseRetriever(config)

    p_encoder.eval()
    q_encoder.eval()

    p_encoder.load_state_dict(torch.load(config["p_encoder_save_path"]))
    q_encoder.load_state_dict(torch.load(config["q_encoder_save_path"]))

    p_encoder.to('cuda')
    q_encoder.to('cuda')

    if os.path.exists(config["corpus_feature_path"]):
        print("found saved features file in", config["corpus_feature_path"])
        print("so we don't inference again.")
        with open(config["corpus_feature_path"], "rb") as f:
            p_outputs = pickle.load(f)
    else:
        print("inference on wiki corpus...")
        p_outputs = []
        for data in tqdm(dataloader):
            data = {k: v.to('cuda') for k, v in data.items()}
            with torch.no_grad():
                p_output = p_encoder(**data)
            p_outputs.append(p_output.cpu())
        p_outputs = torch.cat(p_outputs, dim=0)
        print("done!\nResult features are saved in", config["corpus_feature_path"])
        
        with open(config["corpus_feature_path"], "wb") as f:
            pickle.dump(p_outputs, f)

    # prepare question
    valid_datasets = RetrieverDataset(config, mode="validation")
    valid_loader = DataLoader(valid_datasets, batch_size=16)

    print("preprocessing question and label contexts...")
    q_outputs = []
    label_outputs = []
    for data in valid_loader:
        with torch.no_grad():
            data = [d.to('cuda') for d in data]
            label_output = p_encoder(data[0], data[1], data[2])
            question_output = q_encoder(data[3], data[4], data[5])
        q_outputs.append(question_output.cpu())
        label_outputs.append(label_output.cpu())
    q_outputs = torch.cat(q_outputs, dim=0)
    label_outputs = torch.cat(label_outputs, dim=0)
    print("done!")

    # calculate similarity score.
    scores = torch.matmul(q_outputs, p_outputs.T)
    # and get top-100 results.
    topk_indexes = []
    topk_scores = []
    for score in scores:
        topk_res = torch.topk(score, 100)
        topk_indexes.append(topk_res.indices)
        topk_scores.append(topk_res.values)

    top5 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indexes, 5)
    top20 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indexes, 20)
    top100 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indexes, 100)
    print("top-5 result :", top5)
    print("top-20 result :", top20)
    print("top-100 result :", top100)

def calc_wiki_accuracy(pred_context, label_context, indexes, k):
    correct = 0
    for i, index in enumerate(indexes):
        label = label_context[i]
        for idx in index[:k]: # top-k
            if pred_context[idx].tolist() == label.tolist():
                correct += 1
            
    return correct/len(indexes)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["corpus_feature_path"] = "features/wiki_trun_32.pickle"
    
    main(config)
