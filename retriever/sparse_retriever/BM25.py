import os
import pickle
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, tokenize_fn, corpus_path, pickle_path = "bm25_bert-base.pickle"):
        '''
        tokenize_fn : tokenizer function
        corpus_path : path of corpus in json format.
        pickle_path : path of pretrained vectorizer pickle file. If not exists, it will save new pretrained vectorizer.
        '''
        self.corpus_path = corpus_path
        self.pickle_path = pickle_path
        with open(corpus_path,"r", encoding="utf-8") as f:
            if ".json" in corpus_path:
                wiki = pd.read_json(f).T
            elif ".csv" in corpus_path:
                wiki = pd.read_csv(f)
            else:
                raise AssertionError("corpus경로는 .json이나 .csv 둘 중 하나여야 합니다")

        self.corpus = wiki['text'].unique()
        self.tokenize_fn = tokenize_fn
        print(f"Lengths of unique contexts : {len(self.corpus)}")
        self.bm25 = None

    def get_sparse_embedding(self):
        bm25_path = self.pickle_path
        #pickle_path에 파일이 있으면 불러오고 아니면 bm25를 만들고 저장합니다.
        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            print("bm25 pickle load.")
        else:
            tokenized_corpus = []
            for sentence in self.corpus:
                tokenized_corpus.append(self.tokenize_fn(sentence))
            bm25 = BM25Okapi(tokenized_corpus)
            self.bm25 = bm25
            with open(bm25_path,"wb") as f:
                pickle.dump(self.bm25,f)
            print("bm25 pickle saved.")
        
    def get_relevant_doc(self, query: list, k: int=1):
        '''
        query: list of questions in string format.(list)
        k: number of top object.(int)
        '''
        tokenized_query = [self.tokenize_fn(q) for q in query]
        
        docs = []
        for query in tqdm(tokenized_query):
            top_k = self.bm25.get_top_n(query, self.corpus, k)
            docs.append(top_k)
        
        return docs
    