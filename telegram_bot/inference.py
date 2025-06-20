from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

from src.data.aspect_labels import ASPECT_LABELS

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "src", "models", "aspect_model")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, function_to_apply="sigmoid")

text = "Приложение глючное, цена топлива завышена. Карта не работет. Карта!"

preds = pipe(text)

print("Найденные аспекты:")
for pred in preds[0]:
    label_id = int(pred['label'].replace("LABEL_", ""))  
    aspect_name = ASPECT_LABELS[label_id]
    score = pred['score']
    print(f"{aspect_name}: {score:.2f}")