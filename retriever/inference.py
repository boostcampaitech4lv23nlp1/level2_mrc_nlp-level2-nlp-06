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
    print(f"retriever > test.py > main: Preprocess wikipedia documents")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    wiki_dataset = WikiDataset(config, tokenizer)
    wiki_dataloader = DataLoader(wiki_dataset, batch_size=config["batch_size"])

    ### Load the trained passage and question encoder ###
    print(f"retriever > test.py > main: Load the trained encoders")
    p_encoder = DenseRetriever(config)

    p_encoder.load_state_dict(torch.load(config["p_encoder_load_path"]))

    p_encoder.eval()

    p_encoder = p_encoder.to("cuda")

    ### Get the features of wikipedia documents ###
    if os.path.exists(config["corpus_feature_path"]):
        print(
            f"retriever > test.py > main: Saved features file {config['corpus_feature_path']} is found."
        )
        print("This does not inference again.")
        with open(config["corpus_feature_path"], "rb") as f:
            p_outputs = pickle.load(f)
    else:
        print("retriever > test.py > main: Create the features of wikipedia documents")
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
            f"retriever > test.py > main: Save features in {config['corpus_feature_path']}"
        )
        with open(config["corpus_feature_path"], "wb") as f:
            pickle.dump(p_outputs, f)

    ### Load questions ###
    test_dataset = RetrieverDataset(config, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    print(f"retriever > test.py > main: Preprocessing questions from {config['test_data_path']}")
    q_outputs = []
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            data = [d.to("cuda") for d in data]
            question_output = p_encoder(data[0], data[1], data[2])
        q_outputs.append(question_output.cpu())
    q_outputs = torch.cat(
        q_outputs, dim=0
    )  # Size: (number of questions, dimension of hidden state)

    ### Compute similarity scores between questions and subdocuments ###
    scores = torch.matmul(
        q_outputs, p_outputs.T
    )  # Size: (number of questions, number of subdocuments)

    ### Get top-k subdocuments for each question ###
    topk_indices = []
    topk_scores = []
    for score in scores:
        topk_result = torch.topk(score, config["top_k"])  # Attributes: values, indices
        topk_indices.append(topk_result.indices)
        topk_scores.append(topk_result.values)

    ### Save the pairs of question and top-k subdocuments ###
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 5, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 10, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 20, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 30, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 40, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 50, topk_indices, scores)
    save_topk_file(test_dataset, wiki_dataset, tokenizer, 100, topk_indices, scores)


def save_topk_file(test_dataset, wiki_dataset, tokenizer, topk, topk_indices, scores):
    result = {
        "question": [],
        "subdocument": [],
        "question_id": [],
        "document_id": [],
        "subdocument_id": [],
        "similarity_score": []
    }
    for question_index in range(len(test_dataset)):
        for topk_index in topk_indices[question_index][:topk]:
            token_start_index = 0
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index].sum() == 0:
                token_start_index += 1
            token_end_index = config['max_length'] - 1
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index].sum() == 0:
                token_end_index -= 1
            token_start_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index][0]
            token_end_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index][1]
            result["question"].append(test_dataset.dataset[question_index]["question"])
            result["question_id"].append(question_index)
            result["document_id"].append(
                wiki_dataset.contexts["overflow_to_sample_mapping"][topk_index].item()
            )
            result["subdocument"].append(wiki_dataset.corpus[result["document_id"][-1]][token_start_index:token_end_index + 1])
            result["subdocument_id"].append(topk_index.item())
    pd.DataFrame.from_dict(result).to_csv(config["inference_result_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for retriever.")
    parser.add_argument(
        "--conf", type=str, default="config.yaml", help="config file path(.yaml)"
    )
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
