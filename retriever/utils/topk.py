import torch
import pickle
from tqdm import tqdm
from dataset import WikiDataset
from torch.utils.data import DataLoader


class TOPK:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.wiki_dataset = WikiDataset(config, tokenizer)
        self.wiki_dataloader = DataLoader(self.wiki_dataset, batch_size=config["batch_size"])
        self.p_outputs = None
        self.config = config
        
    def get_passage_outputs(self, p_encoder, epoch):
        p_encoder.eval()
        p_outputs = []
        for data in tqdm(self.wiki_dataloader):
            data = {k: v.to('cuda') for k, v in data.items()}
            with torch.no_grad():
                p_output = p_encoder(**data)
            p_outputs.append(p_output.cpu())
        p_outputs = torch.cat(p_outputs, dim=0)
        # Save corpus features.
        corpus_feature_paths = self.config["corpus_feature_path"]
        if epoch != -1:
            corpus_feature_paths = corpus_feature_paths.replace(".pickle", "")
            corpus_feature_paths = corpus_feature_paths + str(epoch) + ".pickle"
        with open(corpus_feature_paths, "wb") as f:
            pickle.dump(p_outputs, f)
        self.p_outputs = p_outputs
        
        return p_outputs
        
    def get_results(self, p_encoder, dataloader, p_outputs, mode="train"):
        if mode == "test":
            q_outputs = []
            for data in dataloader:
                with torch.no_grad():
                    data = [d.to(self.config["device"]) for d in data]
                    question_output = p_encoder(data[0], data[1], data[2])
                q_outputs.append(question_output.cpu())
            q_outputs = torch.cat(q_outputs, dim=0)
            return q_outputs
        else:
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
            
            return scores, label_outputs
    
    def get_topk_results(self, p_outputs, scores, label_outputs, ks):
        topk_indices = []
        for score in scores:
            topk_res = torch.topk(score, 100)
            topk_indices.append(topk_res.indices)
        top_k_results = []
        for k in ks:
            top_k = self.calc_wiki_accuracy(p_outputs, label_outputs, topk_indices, k)
            top_k_results.append(top_k)
        
        return top_k_results, topk_indices
        
    def calc_wiki_accuracy(self, pred_context, label_context, indexes, k):
        correct = 0
        for i, index in enumerate(indexes):
            label = label_context[i]
            for idx in index[:k]:  # top-k
                if pred_context[idx].tolist() == label.tolist():
                    correct += 1
                    break

        return correct / len(indexes)
    