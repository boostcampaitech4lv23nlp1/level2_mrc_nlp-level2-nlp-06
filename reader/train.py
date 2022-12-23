import logging
import torch
import os
import sys

sys.path.append('/opt/ml/mrccode')
import yaml
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)
from model.select_model import TransformerModel
from dataset import Prepare_train_features, Prepare_validation_features
from utils_qa import check_no_error, postprocess_qa_predictions
import wandb

logger = logging.getLogger(__name__)


def main():
    wandb.init(project="huggingface",name="baselinetest",notes="test")

    set_seed(6)

    with open("arg.yaml","r") as f:
        args = yaml.load(f,Loader = yaml.Loader)

    training_args = TrainingArguments(
            output_dir=args['checkpoint_dir'],
            save_total_limit=2,
            save_steps=args['save_steps'],
            num_train_epochs=args['num_train_epochs'],
            learning_rate=args['learning_rate'],
            per_device_train_batch_size=args['batch_size'],
            per_device_eval_batch_size=args['batch_size'],
            warmup_steps=args['warmup_steps'],
            weight_decay=args['weight_decay'],
            logging_dir='./logs',
            logging_steps=args['eval_steps'],
            evaluation_strategy='no',
            fp16=True,
            do_train=args["do_train"],
            do_eval=args['do_eval'],
            overwrite_output_dir=args['overwrite_output_dir'],
    )

    print(args['model_name'])
    print(f"data is from {args['dataset_name']}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    datasets = load_from_disk(args['dataset_name'])

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        args['config_name'] 
        if args['config_name'] is not None 
        else args['model_name'],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args['tokenizer_name']
        if args['tokenizer_name'] is not None
        else args['model_name'],
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = TransformerModel(args,config)

    if args['model_path'] != None and os.path.exists(args['model_path']):
        model.load_state_dict(torch.load(os.path.join(args['model_path'],"pytorch_model.bin")))

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(args, training_args, datasets, tokenizer, model)


def run_mrc(
    args,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
):

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        args, training_args, datasets, tokenizer
    )
    
    prepare_train_features=Prepare_train_features(tokenizer,max_seq_length,args,pad_on_right)

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=args['preprocessing_num_workers'],
            remove_columns=column_names,
            load_from_cache_file=not args['overwrite_cache'],
        )

    prepare_validation_features=Prepare_validation_features(tokenizer,max_seq_length,args,pad_on_right)

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=args['preprocessing_num_workers'],
            remove_columns=column_names,
            load_from_cache_file=not args['overwrite_cache'],
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

        # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=args['max_answer_length'],
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif args['model_path'] != None and os.path.exists(args['model_path']):
            checkpoint = args['model_path']
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
