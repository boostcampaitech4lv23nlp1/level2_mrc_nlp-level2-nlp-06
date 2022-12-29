from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os


class TfIdfRetrieval:
    def __init__(self, tokenize_fn, corpus_path, pickle_path):
        '''
        tokenize_fn : tokenizer function
        corpus_path : path of corpus in json format.
        pickle_path : path of pretrained vectorizer pickle file. If not exists, it will save new pretrained vectorizer.
        '''
        with open(corpus_path, "r") as f:
            wiki = pd.read_json(f).T
        self.corpus = wiki['text'].unique()
        self.pickle_path = pickle_path
        self.tokenize_fn = tokenize_fn

    def get_sparse_embedding(self):
        if os.path.exists(self.pickle_path):
            print("matrix already exists.")
            with open(self.pickle_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print("so we load saved file from", self.pickle_path)
        else:
            print("make tf-idf features...")
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize_fn,
                ngram_range=(1,2)
            )
            self.vectorizer.fit(self.corpus)
            print("Done!\n save featrues to", self.pickle_path)
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
        self.sp_matrix = self.vectorizer.transform(self.corpus)

    def get_relevant_doc(self, query: list, k: int=1):
        '''
        query: list of questions in string format.(list)
        k: number of top object.(int)
        '''
        query_vec = self.vectorizer.transform(query)
        assert query_vec.max() == 0., "word in question never apeeared in corpus!"
        results = query_vec * self.sp_matrix.T

        docs_scores = []
        docs = []

        for result in tqdm(results):
            sorted_result = np.argsort(-result.data)
            doc_score = result.data[sorted_result]
            doc_ids = result.indices[sorted_result]
            docs_scores.append(doc_score[:k])
            docs.append([self.corpus[idx] for idx in doc_ids[:k]])
        
        return docs_scores, docs
    