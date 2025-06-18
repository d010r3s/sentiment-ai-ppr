import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Retriever:
    def __init__(self):
        config = load_config()
        self.db_path = config["database"]["path"]
        self.model = SentenceTransformer(config["models"]["embedder"])
        self.top_k = config["rag"]["top_k"]
        self.threshold = config["rag"]["similarity_threshold"]

    def retrieve(self, query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT comment, recommendations, embedding FROM feedback WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        if not rows:
            return []
        comments = [row[0] for row in rows]
        recommendations = [row[1] or "" for row in rows]
        embeddings = np.array([pickle.loads(row[2]) for row in rows])
        conn.close()
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        results = [
            (comments[i], recommendations[i], similarities[i])
            for i in top_indices if similarities[i] >= self.threshold
        ]
        if not results and self.threshold > 0.5:
            results = [
                (comments[i], recommendations[i], similarities[i])
                for i in top_indices if similarities[i] >= 0.5
            ]
        return results