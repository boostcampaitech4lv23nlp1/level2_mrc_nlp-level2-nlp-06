import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
from model.model_selection import ModelSelection
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions, generarate_answer
from preprocessing.preprocessor import ExtractionProcessor, GenerationProcessor
from transformers import default_data_collator, TrainingArguments, EvalPrediction
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


## Set seed for random situation
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
## Get config from yaml file
def get_config():
    with open("arg.yaml", "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)
    return args

if __name__ == "__main__":
    set_seed(6)
    torch.cuda.empty_cache()
    
    config = get_config()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    ## Model
    model_selection = ModelSelection(config)
    tokenizer = model_selection.get_tokenizer()
    model = model_selection.get_model()
    model.to(device)
    
    ## Data
    if config["train_type"] == 0:
        preprocessor = ExtractionProcessor(config, tokenizer)
    elif config["train_type"] == 1:
        preprocessor = GenerationProcessor(config, tokenizer)
        
    train_dataset = preprocessor.get_train_dataset()
    eval_dataset = preprocessor.get_eval_dataset()
    eval_examples = preprocessor.get_eval_examples()
    
    
    ## TODO: seperate this part
    ## Training
    if config["train_type"] == 0:
        training_args = TrainingArguments(
            fp16=True,
            do_eval=True,
            do_train=True,
            logging_steps=100,
            logging_dir="./log",
            save_total_limit=2,
            evaluation_strategy='no',
            learning_rate=config["lr"],
            output_dir=config["output_dir"],
            num_train_epochs=config["epoch"],
            weight_decay=config["weight_decay"],
            per_device_eval_batch_size=config["batch_size"],
            per_device_train_batch_size=config["batch_size"],
        )
        
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            post_process_function=preprocessor.post_processing_function,
            compute_metrics=preprocessor.compute_metrics,
        )
    elif config["train_type"] == 1:
        training_args = Seq2SeqTrainingArguments(
            output_dir='outputs', 
            do_train=True, 
            do_eval=True, 
            predict_with_generate=True,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy='epoch',
            save_total_limit=2
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    train_result = trainer.train()
    print(train_result)
    torch.save(model.state_dict(), config["model_save_path"])
    
    ## Evaluation
    if config["train_type"] == 0:      
        metrics = trainer.evaluate()
        print(metrics)
    elif config["train_type"] == 1:
        metrics = trainer.evaluate(
            max_length=config["max_target_length"],
            num_beams=config["num_beams"],
            metric_key_prefix="eval"
        )

        ## TODO: For generation model
        '''''
        for i in np.random.randint(0, len(datasets["validation"]), 5):
        print(generarate_answer(datasets["validation"][int(i)]))
        print("=" * 8)
        '''''