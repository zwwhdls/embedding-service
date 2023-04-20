from abc import ABC
import time

import sentence_transformers
from typing import List

from config import conf
from embeddings.base import Embedding

model_map = {
    "text2vec-large-chinese": conf.text2vec_model_path
}


class HuggingFaceEmbedding(Embedding, ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        if not model_map[self.model_name]:
            print("init begin {}".format(time.asctime(time.localtime(time.time()))))
            self.client = sentence_transformers.SentenceTransformer(self.model_name)
            print("init end {}".format(time.asctime(time.localtime(time.time()))))
        else:
            self.client = sentence_transformers.SentenceTransformer(model_map[self.model_name])

    def embedding_docs(self, texts) -> List[List[float]]:
        print("encode begin {}".format(time.asctime(time.localtime(time.time()))))
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts)
        print("encode end {}".format(time.asctime(time.localtime(time.time()))))
        return embeddings.tolist()

    def embedding_query(self, text) -> List[float]:
        text = text.replace("\n", " ")
        print("encode begin {}".format(time.asctime(time.localtime(time.time()))))
        embedding = self.client.encode(text)
        print("encode end {}".format(time.asctime(time.localtime(time.time()))))
        return embedding.tolist()


if conf.text2vec_model_path:
    text2vect_embedding = HuggingFaceEmbedding("text2vec-large-chinese")
