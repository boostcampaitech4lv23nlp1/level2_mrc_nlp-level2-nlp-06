import os
import yaml
import torch
import pickle
import argparse
import pandas as pd
from utils import TOPK
from model import DenseRetriever
from dataset import RetrieverDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def main(config):
    ### Load wikipedia documents ###
    print(f"retriever > inference.py > main: Preprocess wikipedia documents")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    topk = TOPK(config, tokenizer)
    wiki_dataset = topk.wiki_dataset

    ### Load the trained passage and question encoder ###
    print(f"retriever > inference.py > main: Load the trained encoders")
    p_encoder = DenseRetriever(config)
    p_encoder.load_state_dict(torch.load(config["p_encoder_load_path"]))
    p_encoder.eval()
    p_encoder = p_encoder.to("cuda")

    ### Get the features of wikipedia documents ###
    if os.path.exists(config["corpus_feature_path"]):
        print(
            f"retriever > inference.py > main: Saved features file {config['corpus_feature_path']} is found."
        )
        print("This does not inference again.")
        with open(config["corpus_feature_path"], "rb") as f:
            p_outputs = pickle.load(f)
    else:
        print("retriever > inference.py > main: Create the features of wikipedia documents")
        p_outputs = topk.get_passage_outputs(p_encoder, -1)
        print(
            f"retriever > inference.py > main: Save features in {config['corpus_feature_path']}"
        )

    ### Load questions ###
    test_dataset = RetrieverDataset(config, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    print(f"retriever > inference.py > main: Preprocessing questions from {config['test_data_path']}")
    q_outputs = topk.get_results(p_encoder, test_dataloader, p_outputs, mode="test")

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
    save_topk_file(test_dataset, wiki_dataset, 100, topk_indices, scores)


def save_topk_file(test_dataset, wiki_dataset, topk, topk_indices, scores):
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
            result["subdocument"].append(wiki_dataset.corpus[result["document_id"][-1]].replace("\n", "")[token_start_index:token_end_index + 1])
            result["subdocument_id"].append(topk_index.item())
            result["similarity_score"].append(scores[question_index][topk_index].item())
    pd.DataFrame.from_dict(result).to_csv(f"{config['inference_result_path'][:-4]}-{topk}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for retriever.")
    parser.add_argument(
        "--conf", type=str, default="config.yaml", help="config file path(.yaml)"
    )
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
