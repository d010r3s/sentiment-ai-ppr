import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import f1_score
# from tqdm import tqdm

from aspect_labels import ASPECT_LABELS

DATA_PATH = "sentiment-ai-ppr/datasets/labeled_reviews.csv"


token = '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'
MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"

'''
другие варианты: 
DeepPavlov/rubert-base-cased
cointegrated/rubert-tiny-sentiment-balanced
blanchefort/rubert-base-cased-sentiment
bert-base-multilingual-cased
'''

OUTPUT_DIR = "./aspect-multi-label-model-tiny"


def load_data(path):
    df = pd.read_csv(path)

    texts = df["text"].tolist()
    labels = df[ASPECT_LABELS].values.astype(np.int64)

    return texts, labels


def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = (torch.sigmoid(torch.tensor(probs)) > 0.5).int().numpy()

    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted', zero_division=0)

    return {
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "f1_weighted": weighted_f1
    }


class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


if __name__ == "__main__":
    print("Загрузка данных...")
    texts, labels = load_data(DATA_PATH)

    print(f"Найденные аспекты: {ASPECT_LABELS}")

    print("Токенизация текста...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    print("Разделение на train/val...")
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

    train_dataset = AspectDataset({k: encodings[k][train_idx] for k in encodings}, labels[train_idx])
    val_dataset = AspectDataset({k: encodings[k][val_idx] for k in encodings}, labels[val_idx])

    print("Загрузка модели...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ASPECT_LABELS),
        problem_type="multi_label_classification", 
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Обучение модели...")
    trainer.train()

    print("Сохранение модели...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Модель сохранена в", OUTPUT_DIR)