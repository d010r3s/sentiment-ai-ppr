import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embedder import Embedder
from src.utils.config import load_config


class Retriever:
    def __init__(self):
        config = load_config()
        self.db_path = config["database"]["path"]
        self.embedder = Embedder()
        self.top_k = config["rag"]["top_k"]
        self.threshold = config["rag"]["similarity_threshold"]

    def retrieve(self, query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT comment, recommendations, embedding FROM feedback WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        if not rows:
            print("No records found in feedback.db")
            return []
        comments = [row[0] for row in rows]
        recommendations = [row[1] or "" for row in rows]
        embeddings = np.array([pickle.loads(row[2]) for row in rows])
        conn.close()
        print(f"Retrieved {len(rows)} records from feedback.db")
        query_embedding = self.embedder.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        results = [
            (comments[i], recommendations[i], float(similarities[i]))
            for i in top_indices if similarities[i] >= self.threshold
        ]
        if not results and self.threshold > 0.5:
            results = [
                (comments[i], recommendations[i], float(similarities[i]))
                for i in top_indices if similarities[i] >= 0.5
            ]
        return results


if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Задержка возврата денег")
    print("Retriever results (top 5):")
    for result in results[:5]:
        print(f"Comment: {result[0][:50]}...\nRecommendation: {result[1]}\nSimilarity: {result[2]:.3f}\n")