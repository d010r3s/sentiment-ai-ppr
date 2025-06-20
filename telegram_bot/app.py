from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

from src.data.aspect_labels import ASPECT_LABELS

app = Flask(__name__)

MODEL_PATH = "./aspect-multi-label-model-tiny"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]
    results = []
    for i, score in enumerate(probs):
        if score > 0.5:
            results.append({"aspect": ASPECT_LABELS[i], "score": float(score)})
    return results


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_text(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)