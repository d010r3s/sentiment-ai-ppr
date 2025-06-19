import yaml
import sys
import os
import yaml
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config


class Embedder:
    def __init__(self):
        config = load_config()
        self.model_name = config["models"]["embedder"]
        self.model = SentenceTransformer(self.model_name)

    def encode(self, text):
        if isinstance(text, str):
            return self.model.encode(text)
        elif isinstance(text, list):
            return self.model.encode(text)
        else:
            raise ValueError("Input must be a string or list of strings")