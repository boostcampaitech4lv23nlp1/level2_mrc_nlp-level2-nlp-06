import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from dataset import WikiDataset
from torch.utils.data import DataLoader


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    
class TOPK:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.wiki_dataset = WikiDataset(config, tokenizer)
        self.wiki_dataloader = DataLoader(self.wiki_dataset, batch_size=config["batch_size"])
        self.p_outputs = None
        self.config = config
        
    def get_results(self, p_encoder, epoch, dataloader, inference=True):
        p_encoder.eval()
        if inference:
            p_outputs = []
            for data in tqdm(self.wiki_dataloader):
                data = {k: v.to('cuda') for k, v in data.items()}
                with torch.no_grad():
                    p_output = p_encoder(**data)
                p_outputs.append(p_output.cpu())
            p_outputs = torch.cat(p_outputs, dim=0)
            # Save corpus features.
            if epoch != -1:
                corpus_feature_paths = self.config["corpus_feature_path"].replace(".pickle", "")
                corpus_feature_paths = corpus_feature_paths + str(epoch) + ".pickle"
            with open(corpus_feature_paths, "wb") as f:
                pickle.dump(p_outputs, f)
            self.p_outputs = p_outputs
        else:
            p_outputs = self.p_outputs
            
        q_outputs = []
        label_outputs = []
        for data in dataloader:
            with torch.no_grad():
                data = [d.to(self.config["device"]) for d in data]
                label_output = p_encoder(data[0], data[1], data[2])
                question_output = p_encoder(data[3], data[4], data[5])
            q_outputs.append(question_output.cpu())
            label_outputs.append(label_output.cpu())
        q_outputs = torch.cat(q_outputs, dim=0)
        label_outputs = torch.cat(label_outputs, dim=0)
        
        scores = torch.matmul(q_outputs, p_outputs.T)
        topk_indices = []
        for score in scores:
            topk_res = torch.topk(score, 100)
            topk_indices.append(topk_res.indices)
        top5 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 5)
        top20 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 20)
        top100 = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, 100)
        
        return top5, top20, top100, topk_indices, scores
        
    def calc_wiki_accuracy(self, pred_context, label_context, indexes, k):
        correct = 0
        for i, index in enumerate(indexes):
            label = label_context[i]
            for idx in index[:k]:  # top-k
                if pred_context[idx].tolist() == label.tolist():
                    correct += 1
                    break

        return correct / len(indexes)
    