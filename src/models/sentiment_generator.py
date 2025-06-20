import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.config import load_config
from tqdm import tqdm


class SentimentAnalyzer:
    def __init__(self, model_id: str, batch_size: int = 64, device: str = "auto"):
        """
        Initialize sentiment analyzer with a transformer model.
        Args:
            model_id: Hugging Face model ID or local path.
            batch_size: Batch size for inference.
            device: Device for model (auto, cpu, cuda).
        """
        try:
            self.batch_size = batch_size
            self.device = self._set_device(device)
            is_local = model_id.startswith("/") or model_id.startswith("\\")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=is_local)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, local_files_only=is_local
            ).to(self.device).eval()
        except Exception as e:
            print(f"Error initializing SentimentAnalyzer: {str(e)}")
            raise

    @staticmethod
    def _set_device(device: str) -> str:
        """
        Set device for model inference.
        Args:
            device: Requested device (auto, cpu, cuda).
        Returns:
            Device string.
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def generate_sentiments(self, comments: List[str]) -> List[str]:
        """
        Generate sentiments for a list of comments.
        Args:
            comments: List of comment texts.
        Returns:
            List of sentiments (e.g., 'positive', 'negative', 'neutral').
        """
        try:
            if not comments:
                return []

            preds = []
            for i in tqdm(range(0, len(comments), self.batch_size), desc="Sentiment Analysis", unit="batch"):
                batch_texts = comments[i:i + self.batch_size]
                inputs = self.tokenizer(
                    [str(text) for text in batch_texts],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    batch_preds = [self.model.config.id2label[p] for p in logits.argmax(-1).tolist()]
                    preds.extend(batch_preds)

            return [pred.lower() for pred in preds]

        except Exception as e:
            print(f"Error generating sentiments: {str(e)}")
            return ['neutral'] * len(comments)


def generate_sentiments(comments: List[str]) -> List[str]:
    """
    Generate sentiments for comments using config settings.
    Args:
        comments: List of comment texts.
    Returns:
        List of sentiments.
    """
    config = load_config()
    model_id = config["models"].get("sentiment", "ai-forever/sbert_large_nlu_ru")
    batch_size = config["models"].get("sentiment_batch_size", 64)
    device = config["models"].get("sentiment_device", "auto")

    analyzer = SentimentAnalyzer(model_id=model_id, batch_size=batch_size, device=device)
    return analyzer.generate_sentiments(comments)