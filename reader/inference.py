import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset
from datasets import load_from_disk
from model.model_selection import ModelSelection


## Get config from yaml file
def get_config():
    with open("test_arg.yaml", "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)
    return args


if __name__ == "__main__":
    
    config = get_config()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    ## Get Data
    data = pd.read_csv(config["retriver_path"])
    
    ## Store n_best_answer
    store = [[] for _ in range(len(data))]
 
    ## Load Model
    model_selection = ModelSelection(config)
    model = model_selection.get_model()
    tokenizer = model_selection.get_tokenizer()
    model.load_state_dict(torch.load(config["model_path"]))
    model.to(device)
    model.eval()
    
    ## Inference
    for i in tqdm(range(len(data))):
        now_data = data.iloc[i]
        question_id = now_data["question_id"]
        
        tokenized_data = tokenizer(
            now_data["question" if config["position"] else "subdocument"],
            now_data["subdocument" if config["position"] else "question"],
            truncation="only_second" if config["position"] else "only_first",
            max_length=config["mx_token_length"],
            return_offsets_mapping=True,
            return_token_type_ids=False if "roberta" in config["model_name"] else True,
            padding="max_length" if True else False,
            return_tensors="pt",
        )
        
        offset_mapping = tokenized_data["offset_mapping"][0]
        
        ## Only for Roberta
        if "roberta" in config["model_name"]:
            token_type_ids = None
        else:
            token_type_ids = tokenized_data["token_type_ids"]
        
        with torch.no_grad():
            output = model(
                input_ids=tokenized_data["input_ids"].to(device),
                attention_mask=tokenized_data["attention_mask"].to(device),
                token_type_ids=token_type_ids.to(device),
                )

        start_logits = output["start_logits"][0]
        end_logits = output["end_logits"][0]
        
        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        
        ## Update the Score
        score = start_logits[0] + end_logits[0]
        if config["min_score"] > score:
            null_prediction = {
                "offsets": (0, 0),
                "score": score,
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
            }
        
        start_indexes = np.argsort(start_logits)[-1 : -config["n_best_size"]-1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -config["n_best_size"]-1 : -1].tolist()
        
        for start_index in start_indexes:
            if not start_index: 
                continue
            for end_index in end_indexes:
                if not end_index:
                    continue
                # out-of-scope answers는 고려하지 않습니다.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > config["max_answer_length"]
                ):
                    continue
                
                store[question_id].append(
                    {
                        "offsets": (
                            offset_mapping[start_index][0],
                            offset_mapping[end_index][1],
                        ),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                        "context": now_data["subdocument"]
                    }
                )
        
    ## Get Predictions & Answers
    predictions = []
    for item in store:
        if not item: continue
        predictions.append(
            sorted(
                item, key=lambda x: x["score"], reverse=True
            )[:config["n_best_size"]]
        )
    
    answer = []
    for prediction in predictions:
        scores = np.array([pred.pop("score") for pred in prediction])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        for prob, pred in zip(probs, prediction):
            offsets = pred.pop("offsets")
            pred["text"] = pred["context"][int(offsets[0]) : int(offsets[1])]
            pred["probability"] = prob
        
        i = 0
        while (
            prediction[i]["text"] == "" or 
            len(prediction[i]["text"]) > config["max_answer_length"] #or
            #"\n" in prediction[i]["text"] or
            #"." in prediction[i]["text"] or
            #"," in prediction[i]["text"]
        ):
            if i == len(prediction) - 1: 
                break
            i += 1
        answer.append(prediction[i]["text"])
    
    if config["eval_or_test"] == 0:
        test_data = load_from_disk("/opt/ml/input/data/train_dataset")["validation"]
    elif config["eval_or_test"] == 1:
        test_data = load_from_disk("/opt/ml/input/data/test_dataset")["validation"]
    
    test_id = test_data["id"]
    question = test_data["question"]
    
    result = OrderedDict()
    for i in range(len(answer)):
        result[test_id[i]] = answer[i]
        
    prediction_file = os.path.join("/opt/ml/results/", "predictions.json")  
    with open(prediction_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(result, indent=4, ensure_ascii=False) + "\n"
        )
    
    
    
    
    
    
    