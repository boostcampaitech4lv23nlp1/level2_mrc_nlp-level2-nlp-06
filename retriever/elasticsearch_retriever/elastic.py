import pandas as pd
import json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


class ESRetriever:
    def __init__(self):
        self.es = None

    def connect(self,port=9200):
        self.es = Elasticsearch([{"host":"localhost","port":port,"scheme":"http"}])
        print("Connected" if self.es.ping() else "Not connected")
    
    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name)
    
    def get_indices_name(self):
        res = self.es.indices.get(index="*")
        return res

    def create_index(self, index_name, similarity,corpus_path):
        '''
            Args:
                index_name: 만들고자 하는 인덱스 이름 (:obj:`str`)
                similarity: 설정할 검색 방법 이름 (:obj:`str`)
                corpus_path: 찾아올 대상이 되는 large corpus 파일 경로 (:obj:`str`)
        '''
        with open("body.json","r") as f:
            body = json.load(f)
        
        assert similarity in [
            "BM25_similarity",
            "DFR_similarity",
            "DFI_similarity",
            "IB_similarity",
            "LMDirichlet_similarity",
            "LMJelinekMercer_similarity"], f"invalid similarity"

        body["mappings"]["properties"]["text"]["similarity"]=similarity
        self.es.indices.create(index=index_name, body=body)

        df = pd.read_csv(corpus_path)
        docs = []
        for i in tqdm(range(len(df))):
            if not isinstance(df.iloc[i]["text"],str):
                continue
            docs.append({
                "_id": i,
                '_index': index_name,
                '_source': {
                    "id": df.iloc[i]["id"],
                    "text": df.iloc[i]["text"]
                    }
            })
        helpers.bulk(self.es, docs)

    def get_relevant_doc(self,index_name,topk,data_path,output_path):         
        '''
        Args:
            index_name: 쿼리를 날리고자 하는 인덱스 (:obj:`str`)
            topk: 하나의 question당 찾아오는 passage 개수 (:obj:`int`)
            data_path: question이라는 column이 들어가 있는 csv파일 경로 (:obj:`str`)
            output_path: 결과를 저장할 경로(.csv) (:obj:`str`)
        '''
        df1 = pd.read_csv(data_path)
        output_df1 = pd.DataFrame(columns=["question","subdocument","similarity_score","question_id"])
        question=[]
        subdocument=[]
        similarity_score=[]
        question_id=[]
        for i in tqdm(range(len(df1))):
            res = self.es.search(index=index_name, body={
                "query": { "match": {"text":df1.iloc[i]["question"]} },
                "size": topk
            })
            for hit in res['hits']['hits']:
                question.append(df1.iloc[i]["question"])
                subdocument.append(hit["_source"]["text"])
                similarity_score.append(hit["_score"])
                question_id.append(i)
        output_df1["question"]=question
        output_df1["subdocument"]=subdocument
        output_df1["similarity_score"]=similarity_score
        output_df1["question_id"]=question_id

        print(len(output_df1))
        output_df1.to_csv(output_path)


