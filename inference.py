from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

from aspect_labels import ASPECT_LABELS


model_path = "./aspect-multi-label-model-tiny"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, function_to_apply="sigmoid")

text = "Приложение глючное, цена топлива завышена. Карта не работет. Карта!"

preds = pipe(text)

print("Найденные аспекты:")
for pred in preds[0]:
    label_id = int(pred['label'].replace("LABEL_", ""))  
    aspect_name = ASPECT_LABELS[label_id]
    score = pred['score']
    print(f"{aspect_name}: {score:.2f}")