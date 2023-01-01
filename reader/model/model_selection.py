from model.models import ExtractionModel
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer


class ModelSelection():
    def __init__(self, config):
        self.config = config
        
        if config["train_type"] == 0:
            model_config = AutoConfig.from_pretrained(config["model_name"])
            self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
            self.model = ExtractionModel(config, model_config)
            if config["retrain"] == 1:
                self.model.load_state_dict(torch.load(config["retrain_path"]))
                
        ## TODO: Generation Model
        elif config["train_type"] == 1:
            pass
        
    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer