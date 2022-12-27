from model.models import ExtractionModel
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer


class ModelSelection():
    def __init__(self, config):
        self.config = config
        
        if config["train_type"] == 0:
            model_config = AutoConfig.from_pretrained(config["model_name"])
            self.model = ExtractionModel(config, model_config)
        
        ## TODO: Generation Model
        elif config["train_type"] == 1:
            pass
        
    def get_model(self): return self.model