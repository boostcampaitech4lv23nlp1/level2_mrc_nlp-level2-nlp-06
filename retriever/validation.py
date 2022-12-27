import os
import yaml
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from model import DenseRetriever
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import RetrieverDataset, WikiDataset


def main(config):
    ### Load wikipedia documents ###
    print(f"retriever > validation.py > main: Preprocess wikipedia documents")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    wiki_dataset = WikiDataset(config, tokenizer)
    wiki_dataloader = DataLoader(wiki_dataset, batch_size=config["batch_size"])

    ### Load the trained passage and question encoder ###
    print(f"retriever > validation.py > main: Load the trained encoders")
    p_encoder = DenseRetriever(config)
    q_encoder = DenseRetriever(config)

    p_encoder.load_state_dict(torch.load(config["p_encoder_save_path"]))
    q_encoder.load_state_dict(torch.load(config["q_encoder_save_path"]))

    p_encoder.eval()
    q_encoder.eval()

    p_encoder = p_encoder.to("cuda")
    q_encoder = q_encoder.to("cuda")

    ### Get the features of wikipedia documents ###
    if os.path.exists(config["corpus_feature_path"]):
        print(
            f"retriever > validation.py > main: Saved features file {config['corpus_feature_path']} is found."
        )
        print("This does not inference again.")
        with open(config["corpus_feature_path"], "rb") as f:
            p_outputs = pickle.load(f)
    else:
        print("retriever > validation.py > main: Create the features of wikipedia documents")
        p_outputs = []
        for data in tqdm(wiki_dataloader):
            data = {k: v.to("cuda") for k, v in data.items()}
            with torch.no_grad():
                p_output = p_encoder(**data)
            p_outputs.append(p_output.cpu())
        p_outputs = torch.cat(
            p_outputs, dim=0
        )  # Size: (number of subdocuments, dimension of hidden state)
        print(
            f"retriever > validation.py > main: Save features in {config['corpus_feature_path']}"
        )
        with open(config["corpus_feature_path"], "wb") as f:
            pickle.dump(p_outputs, f)

    ### Load questions ###
    valid_dataset = RetrieverDataset(config, mode="validation")
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])

    print(
        f"retriever > validation.py > main: Preprocessing questions from the validation set of {config['train_data_path']}"
    )
    q_outputs = []
    label_outputs = []
    for data in valid_dataloader:
        with torch.no_grad():
            data = [d.to("cuda") for d in data]
            label_output = p_encoder(data[0], data[1], data[2])
            question_output = q_encoder(data[3], data[4], data[5])
        q_outputs.append(question_output.cpu())
        label_outputs.append(label_output.cpu())
    q_outputs = torch.cat(
        q_outputs, dim=0
    )  # Size: (number of questions, dimension of hidden state)
    label_outputs = torch.cat(
        label_outputs, dim=0
    )  # Size: (number of questions, dimension of hidden state)

    ### Compute similarity scores between questions and subdocuments ###
    scores = torch.matmul(q_outputs, p_outputs.T)
    topk_indices = []
    topk_scores = []
    for score in scores:
        topk_res = torch.topk(score, 100)
        topk_indices.append(topk_res.indices)
        topk_scores.append(topk_res.values)

    top5 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 5)
    top20 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 20)
    top100 = calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 100)
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

    return correct / len(indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for retriever.")
    parser.add_argument(
        "--conf", type=str, default="config.yaml", help="config file path(.yaml)"
    )
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
