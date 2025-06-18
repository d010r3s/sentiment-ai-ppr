# src/models/recommender.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import textwrap


class Recommender:
    def __init__(self, model_id="IlyaGusev/saiga_mistral_7b_lora", retriever=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ).eval()
        self.streamer = TextStreamer(self.tokenizer)
        self.retriever = retriever

    def prompt(self, comment):
        company_ctx = textwrap.dedent("""
            Компания: «Передовые Платёжные Решения» (ППР)
            Тип бизнеса: финтех-оператор B2B
            Клиенты: корпоративные автопарки и командировочные службы (≈80k компаний)
            Услуги: оплата топлива, штрафы, парковки, платные дороги, отчётность по расходам
            Цель: снижать негатив клиентов, улучшать поддержку, ускорять возвраты, держать тарифы конкурентными
        """).strip()
        fewshot = textwrap.dedent("""
            ### Отзывы:
            - Слишком долго оформляют возврат.
            - Операторы не помогают решить вопрос.
            ### Рекомендации:
            1. Сократить срок возврата средств до 5 дней.
            2. Ввести KPI операторов: не менее 80 % решённых обращений.
        """).strip()
        forbid = "Не предлагай уходить к конкурентам. Не предлагай закрыть счёт или заменить компанию."
        system = (
            f"Ты — эксперт по клиентскому опыту. Я — менеджер компании ППР. "
            f"Дай до 10 конкретных рекомендаций, как снизить негатив. "
            f"{forbid} Формат ответа: нумерованный список 4–6 пунктов.\n\n"
            f"Контекст компании и клиентов: \n{company_ctx}"
        )
        similar_text = ""
        if self.retriever:
            similar = self.retriever.retrieve(comment)
            if similar:
                similar_text = "\n".join(
                    f"- Comment: {sim[0]}\n  Recommendation: {sim[1]}" for sim in similar
                )
            else:
                system += "\n\nНет похожих отзывов. Сосредоточься на новом отзыве и контексте компании."
        user = f"Новый отзыв: {comment}\nПохожие отзывы и рекомендации: \n{similar_text}"
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{fewshot}\n\n{user}\n\n### Рекомендации: [/INST]"

    def generate(self, comment):
        inp = self.tokenizer(self.prompt(comment), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inp,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                max_new_tokens=256,
            )
        return self.tokenizer.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def generate_batch(self, comments, batch_size=8):
        outputs = []
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            for comment in batch:
                outputs.append(self.generate(comment))
        return outputs