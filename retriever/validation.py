import os
import yaml
import torch
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from utils import TOPK
from model import DenseRetriever
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import RetrieverDataset, WikiDataset


def main(config):
    ### Load wikipedia documents ###
    print(f"retriever > validation.py > main: Preprocess wikipedia documents")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    topk = TOPK(config, tokenizer)

    ### Load the trained passage and question encoder ###
    print(f"retriever > validation.py > main: Load the trained encoders")
    p_encoder = DenseRetriever(config)
    p_encoder.load_state_dict(torch.load(config["p_encoder_load_path"]))
    p_encoder.eval()
    p_encoder = p_encoder.to("cuda")

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
        p_outputs = topk.get_passage_outputs(p_encoder, -1)
        print(
            f"retriever > validation.py > main: Save features in {config['corpus_feature_path']}"
        )

    ### Load questions ###
    valid_dataset = RetrieverDataset(config, mode="validation")
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    
    ks = [5, 10, 20, 50, 100]
    scores, label_outputs = topk.get_results(p_encoder, valid_dataloader, p_outputs)
    [top5, top10, top20, top50, top100], topk_indices = topk.get_topk_results(p_outputs, scores, label_outputs, ks)
    
    print("top-5 result :", top5)
    print("top-10 result :", top10)
    print("top-20 result :", top20)
    print("top-50 result :", top50)
    print("top-100 result :", top100)

    ### Save the pairs of question and top-k subdocuments ###
    save_topk_file(valid_dataset, topk.wiki_dataset, tokenizer, 100, topk_indices, scores)

def save_topk_file(valid_dataset, wiki_dataset, tokenizer, topk, topk_indices, scores):
    result = {
        "question": [],
        "answer_document":[],
        "answer_document_id": [],
        "answer_text": [],
        "subdocument": [],
        "question_id": [],
        "document_id": [],
        "subdocument_id": [],
        "similarity_score": []
    }
    real_question_index = -1
    before_question = None
    for question_index in range(len(topk_indices)):
        question = tokenizer.decode(valid_dataset[question_index][3], skip_special_tokens=True)
        if question != before_question:
            real_question_index += 1
        else:
            continue
        before_question = question

        for topk_index in topk_indices[question_index][:topk]:
            token_start_index = 0
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index].sum() == 0:
                token_start_index += 1
            token_end_index = config['max_length'] - 1
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index].sum() == 0:
                token_end_index -= 1
            token_start_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index][0]
            token_end_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index][1]

            result["question"].append(valid_dataset.dataset[real_question_index]["question"])
            result["question_id"].append(real_question_index)
            result["answer_document_id"].append(valid_dataset.dataset[real_question_index]["document_id"])
            result["answer_document"].append(valid_dataset.dataset[real_question_index]["context"].replace("\\n", ""))

            result["document_id"].append(
                wiki_dataset.contexts["overflow_to_sample_mapping"][topk_index].item()
            )
            result["answer_text"].append(valid_dataset.dataset[real_question_index]["answers"]["text"][0])

            result["subdocument"].append(wiki_dataset.corpus[result["document_id"][-1]][token_start_index:token_end_index + 1].replace("\n", ""))
            result["subdocument_id"].append(topk_index.item())
            result["similarity_score"].append(scores[question_index][topk_index].item())
    pd.DataFrame.from_dict(result).to_csv(f"{config['validation_result_path'][:-4]}-{topk}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for retriever.")
    parser.add_argument(
        "--conf", type=str, default="config.yaml", help="config file path(.yaml)"
    )
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)
