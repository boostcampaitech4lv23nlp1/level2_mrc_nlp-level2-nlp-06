from transformers import AutoTokenizer
from datasets import load_from_disk
from tf_idf import TfIdfRetrieval
from BM25 import BM25Retriever
import pandas as pd


def main():
    corpus_path = "../../../ODQA/data/wikipedia_documents.json"
    pickle_path = "tf-idf_bert-base.pickle"
    tokenizer_name = "klue/bert-base"
    validation_path = "../../../ODQA/data/train_dataset"
    result_path = "tf-idf_valid_result.csv"
    sparse = "tf-idf"
    top_k = 3
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
            df_dict["question_id"].append(valid_data["question_id"][i])
            df_dict["question"].append(valid_data["question"][i])
            df_dict["subdocument"].append(doc)
            
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(result_path)
    
    
if __name__=="__main__":
    main()
    