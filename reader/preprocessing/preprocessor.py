import nltk
import numpy as np
import pandas as pd
from utils_qa import postprocess_qa_predictions
from datasets import DatasetDict, load_from_disk, load_metric, load_dataset
from transformers import default_data_collator, AutoTokenizer, EvalPrediction


class ExtractionProcessor():
    def __init__(self, config, tokenizer):
        
        self.config = config
        
        self.tokenizer = tokenizer
        
        if config["dataset"] == None:
            self.all_data = load_from_disk("/opt/ml/input/data/train_dataset")
        else:
            self.all_data = load_dataset(config["dataset"])
        
        self.train_data = self.all_data["train"]
        self.eval_data = self.all_data["validation"]
        
        if config["num_sample"] != -1:
            self.train_data = self.train_data.select(range(config["num_sample"]))
            self.eval_data = self.eval_data.select(range(config["num_sample"]))
        
        self.train_flag = True
        if self.train_flag:
            self.column_names = self.train_data.column_names
        else:
            self.column_names = self.eval_data.column_names
            
        self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]

        self.position = config["position"]
        
        self.train_dataset = self.train_data.map(
            self.prepare_train_features,
            batched=True,
            num_proc=config["num_proc"],
            remove_columns=self.column_names,
            load_from_cache_file=True, 
        )
        
        self.eval_examples = self.eval_data
        self.eval_dataset = self.eval_data.map(
            self.prepare_validation_features,
            batched=True,
            num_proc=config["num_proc"],
            remove_columns=self.column_names,
            load_from_cache_file=True,
        )
    
    
    def prepare_train_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.position else self.context_column_name],
            examples[self.context_column_name if self.position else self.question_column_name],
            truncation="only_second" if self.position else "only_first",
            max_length=self.config["mx_token_length"],
            stride=self.config["stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if "roberta" in self.config["model_name"] else True,
            padding="max_length" if True else False,
        )
        
        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        overflow_to_sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples = self.tokenized_examples_sttend_pos(tokenized_examples, overflow_to_sample_mapping, offset_mapping, examples)
        
        return tokenized_examples
    
    #tokenized 결과가 eval_loss를 계산하기 위해 prepare_valid_loss에서도 사용되어야 하므로 재사용을 위해 함수로 분리했습니다
    def tokenized_examples_sttend_pos(self, tokenized_examples, overflow_to_sample_mapping, offset_mapping, examples):
        # 정답지를 만들기 위한 리스트
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # 해당 example에 해당하는 sequence를 찾음
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # sequence가 속하는 example을 찾는다
            example_index = overflow_to_sample_mapping[i]
            answers = examples["answers"][example_index]
            
            # 텍스트에서 answer의 시작점, 끝점
            answer_start_offset = answers["answer_start"][0]
            answer_end_offset = answer_start_offset + len(answers["text"][0])
            
            # 텍스트에서 현재 span의 시작 토큰 인덱스
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # 텍스트에서 현재 span 끝 토큰 인덱스
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # answer가 현재 span을 벗어났는지 체크
            if not (
                offsets[token_start_index][0] <= answer_start_offset
                and offsets[token_end_index][1] >= answer_end_offset
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 옮김
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= answer_start_offset
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= answer_end_offset:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            
        return tokenized_examples


    def prepare_validation_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.position else self.context_column_name],
            examples[self.context_column_name if self.position else self.question_column_name],
            truncation="only_second" if self.position else "only_first",
            max_length=self.config["mx_token_length"],
            stride=self.config["stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if "roberta" in self.config["model_name"] else True,
            padding="max_length" if True else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples['offset_mapping']
        tokenized_examples = self.tokenized_examples_sttend_pos(tokenized_examples, sample_mapping, offset_mapping, examples)

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    
    def post_processing_function(self, examples, features, predictions):
        # Post-processing: original context에서 start logit과 end logit을 matching
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=self.config["n_best_size"],
            max_answer_length=self.config["max_answer_length"],
            null_score_diff_threshold=self.config["null_score_diff_threshold"],
            output_dir=self.config["output_dir"],
        )

        # Metric을 계산할 수 있는 format으로 수정
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        references = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in self.eval_data
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


    def compute_metrics(self, p: EvalPrediction):
        metric = load_metric("squad")
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    
    def get_train_dataset(self): return self.train_dataset
    def get_eval_dataset(self): return self.eval_dataset
    def get_train_examples(self): return self.train_data
    def get_eval_examples(self): return self.eval_data


class GenerationProcessor():
    def __init__(self, config, tokenizer):
        self.config = config
        
        self.tokenizer = tokenizer

        ## TODO: for other dataset
        #self.all_data = load_from_disk("/opt/ml/input/data/train_dataset")
        self.all_data = load_dataset(config["dataset"])
        
        self.train_data = self.all_data["train"]
        self.eval_data = self.all_data["validation"]
        
        ## Activate this only you are testing the code
        self.train_data = self.train_data.select(range(config["num_sample"]))
        self.eval_data = self.eval_data.select(range(config["num_sample"]))

        self.column_names = self.train_data.column_names

        self.train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.config["num_proc"],
            remove_columns=column_names,
            load_from_cache_file=False,
        )
        
        self.eval_dataset = eval_examples.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.config["num_proc"],
            remove_columns=column_names,
            load_from_cache_file=False,
        )


    def preprocess_function(self, examples):
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=padding,
            truncation=True
        )

        # targets(label)을 위해 tokenizer 설정
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"] 
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs
    
    
    def postprocess_text(self, preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
            
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"].select(range(max_val_samples)))]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(max_val_samples))]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result
    
    
    def get_train_dataset(self): return self.train_dataset
    def get_eval_dataset(self): return self.eval_dataset
    def get_train_data(self): return self.train_data
    def get_eval_data(self): return self.eval_data 