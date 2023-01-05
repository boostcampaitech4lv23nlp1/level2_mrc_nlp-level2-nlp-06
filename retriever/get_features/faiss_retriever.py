import faiss


class FaissRetriever:
    def __init__(self, p_embs, num_clusters=8) -> None:
        self.num_clusters = num_clusters
        self.p_embs = p_embs
        self.indexer = None
        self.fitting()
    
    def fitting(self):
        emb_dim = self.p_embs.shape[-1]
        quantizer = faiss.IndexFlatL2(emb_dim)
        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer,
            quantizer.d,
            self.num_clusters,
            faiss.METRIC_L2
        )
        self.indexer.train(self.p_embs)
        self.indexer.add(self.p_embs)
        
    def topk(self, q_embs, k=5):
        distance, index = self.indexer.search(q_embs, k)
        return distance, index
    