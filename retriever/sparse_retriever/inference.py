from transformers import AutoTokenizer
from datasets import load_from_disk
from tf_idf import TfIdfRetrieval
from BM25 import BM25Retriever
import pandas as pd
import argparse
import yaml


def main(config):
    corpus_path = config["corpus_path"]
    pickle_path = config["pickle_path"]
    tokenizer_name = config["tokenizer_name"]
    validation_path = config["validation_path"]
    result_path = config["result_path"]
    sparse = config["sparse"]
    top_k = config["top_k"]
    assert sparse in ["BM25", "tf-idf"], print("sparse must be in ['BM25', 'tf-idf']")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenize_fn = tokenizer.tokenize
    
    valid_data = load_from_disk(validation_path)["validation"]
    questions = valid_data["question"]
    
    if sparse == "BM25":
        print("load BM25Retriever...")
        SparseRetrieval = BM25Retriever(tokenize_fn, corpus_path, pickle_path)
        SparseRetrieval.get_sparse_embedding()
        print(f"get top-{top_k} relevant documents...")
        docs = SparseRetrieval.get_relevant_doc(query=questions, k=top_k)
    elif sparse == "tf-idf":
        print("load TfIdfRetriever...")
        SparseRetrieval = TfIdfRetrieval(tokenize_fn, corpus_path, pickle_path)
        SparseRetrieval.get_sparse_embedding()
        print(f"get top-{top_k} relevant documents...")
        docs_scores, docs = SparseRetrieval.get_relevant_doc(query=questions, k=top_k)
    
    df_dict = {"question_id": [], "question": [], "subdocument": []}
    
    for i, result in enumerate(docs):
        for doc in result:
            df_dict["question_id"].append(i)
            df_dict["question"].append(valid_data["question"][i])
            df_dict["subdocument"].append(doc)
            
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(result_path, index=False)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='config for retriever.')
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main(config)
    