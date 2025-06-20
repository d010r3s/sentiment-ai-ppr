# aspect_generator.py
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from src.utils.config import load_config
from src.data.aspect_labels import ASPECT_LABELS


class AspectAnalyzer:
    def __init__(self, model_id: str, batch_size: int = 64, device: str = "auto"):
        self.batch_size = batch_size
        self.device = self._set_device(device)
        is_local = model_id.startswith("/") or model_id.startswith("\\")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=is_local)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, local_files_only=is_local
            ).to(self.device).eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    @staticmethod
    def _set_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def generate_aspects(self, comments: List[str]) -> List[List[str]]:
        if not comments:
            return []

        results = []
        # Clean comments: convert to strings and filter out invalid entries
        clean_comments = [str(comment) if comment is not None else "" for comment in comments]

        for i in tqdm(range(0, len(clean_comments), self.batch_size), desc="Aspect Extraction", unit="batch"):
            batch_texts = clean_comments[i:i + self.batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.sigmoid(logits).cpu().numpy()

                for prob_vec in probs:
                    aspects = [ASPECT_LABELS[i] for i, p in enumerate(prob_vec) if p > 0.3]
                    results.append(aspects)  # Always append a list, even if empty
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Append empty lists for failed batch to maintain alignment
                results.extend([[] for _ in batch_texts])

        return results


def generate_aspects(comments: List[str]) -> List[List[str]]:
    """
    Generate aspects using config settings.
    """
    config = load_config()
    model_id = config["models"].get("aspect", "src/models/aspect_model")
    batch_size = config["models"].get("aspect_batch_size", 16)
    device = config["models"].get("aspect_device", "auto")

    try:
        analyzer = AspectAnalyzer(model_id=model_id, batch_size=batch_size, device=device)
        return analyzer.generate_aspects(comments)
    except Exception as e:
        print(f"Aspect generation failed: {e}")
        return [[] for _ in comments]  # Return empty lists for all comments