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
    print("done!")
    wiki_dataloader = DataLoader(wiki_dataset, batch_size=config["batch_size"])

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
            question_output = p_encoder(data[3], data[4], data[5])
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

    ### Save the pairs of question and top-k subdocuments ###
    result = {
        "question": [],
        "answer_document":[],
        "answer_document_id": [],
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
        before_question = question
        for topk_index in topk_indices[question_index]:
            token_start_index = 0
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index].sum() == 0:
                token_start_index += 1
            token_end_index = config['max_length'] - 1
            while wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index].sum() == 0:
                token_end_index -= 1
            token_start_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_start_index][0]
            token_end_index = wiki_dataset.contexts["offset_mapping"][topk_index][token_end_index][1]
            
            # 중복된 질문을 해결하기 위한 코드
            result["question"].append(valid_dataset.dataset[real_question_index]["question"])
            result["question_id"].append(real_question_index)
            
            result["answer_document"].append(valid_dataset.dataset[real_question_index]["context"])
            result["document_id"].append(
                wiki_dataset.contexts["overflow_to_sample_mapping"][topk_index].item()
            )
            result["subdocument"].append(wiki_dataset.corpus[result["document_id"][-1]][token_start_index:token_end_index + 1])
            result["subdocument_id"].append(topk_index.item())
            result["similarity_score"].append(scores[question_index][topk_index].item())
    pd.DataFrame.from_dict(result).to_csv(config["validation_result_path"])


def calc_wiki_accuracy(pred_context, label_context, indexes, k):
    correct = 0
    for i, index in enumerate(indexes):
        label = label_context[i]
        for idx in index[:k]:  # top-k
            if pred_context[idx].tolist() == label.tolist():
                correct += 1
                break

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
