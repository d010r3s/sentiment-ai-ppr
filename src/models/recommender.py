# src/models/recommender.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import textwrap
from src.utils.config import load_config


class Recommender:
    def __init__(self, model_id: str = None, retriever=None):
        """
        Initialize recommender with configurable generation parameters.

        Args:
            model_id: Hugging Face model ID (defaults to config value)
            retriever: Optional retriever for similar past cases
        """
        # Load configuration
        config = load_config()
        model_config = config["models"]["recommender"]
        self.use_recommender = config["models"]['use_recommender']

        # Hardcoded recommendations for negative comments per aspect
        self.hardcoded_recommendations = {
            "приложение": [
                "1. Обновить приложение для устранения сбоев.",
                "2. Упростить интерфейс для удобства.",
                "3. Добавить оффлайн-режим.",
                "4. Ускорить загрузку данных.",
                "5. Провести UX-тестирование."
            ],
            "топливо": [
                "1. Проверить точность расчетов топлива.",
                "2. Ускорить обработку топливных транзакций.",
                "3. Предоставить прозрачные отчеты по расходам.",
                "4. Ввести скидки на топливо для лояльных клиентов.",
                "5. Улучшить интеграцию с топливными станциями."
            ],
            "карта": [
                "1. Ускорить выпуск новых карт.",
                "2. Улучшить процесс замены утерянных карт.",
                "3. Добавить поддержку бесконтактных платежей.",
                "4. Устранить ошибки при активации карт.",
                "5. Предоставить инструкции по использованию карт."
            ],
            "поддержка": [
                "1. Увеличить штат операторов поддержки.",
                "2. Ввести KPI для скорости ответа.",
                "3. Обучить персонал решению сложных вопросов.",
                "4. Добавить чат-бот для базовых запросов.",
                "5. Улучшить доступность горячей линии."
            ],
            "интерфейс": [
                "1. Провести аудит интерфейса.",
                "2. Упростить навигацию в приложении.",
                "3. Добавить подсказки для новых пользователей.",
                "4. Устранить визуальные баги.",
                "5. Сделать шрифты более читаемыми."
            ],
            "отчет": [
                "1. Ускорить генерацию отчетов.",
                "2. Добавить настраиваемые шаблоны отчетов.",
                "3. Устранить ошибки в данных отчетов.",
                "4. Ввести автоматическую отправку отчетов.",
                "5. Улучшить визуализацию данных."
            ],
            "эвакуатор": [
                "1. Сократить время ожидания эвакуатора.",
                "2. Улучшить координацию с водителями.",
                "3. Предоставить точное время прибытия.",
                "4. Ввести компенсацию за задержки.",
                "5. Увеличить число доступных эвакуаторов."
            ],
            "цена": [
                "1. Ввести гибкие тарифы для малого бизнеса.",
                "2. Предоставить прозрачное ценообразование.",
                "3. Уменьшить комиссии за транзакции.",
                "4. Ввести программы лояльности.",
                "5. Сравнить цены с конкурентами."
            ],
            "страховка": [
                "1. Ускорить обработку страховых случаев.",
                "2. Упростить подачу заявлений на страховку.",
                "3. Улучшить коммуникацию по статусу дел.",
                "4. Предоставить больше страховых опций.",
                "5. Снизить стоимость страховых пакетов."
            ],
            "шины_диски": [
                "1. Увеличить ассортимент шин и дисков.",
                "2. Ускорить доставку заказов.",
                "3. Предоставить гарантию на продукцию.",
                "4. Улучшить качество консультаций.",
                "5. Ввести скидки на сезонные товары."
            ]
        }

        if self.use_recommender:
            # Set parameters with config fallbacks
            self.model_id = model_id or model_config["model_id"]

            # Generation parameters with config fallbacks
            self.gen_params = {
                'temperature': model_config["temperature"],
                'top_p': model_config["top_p"],
                'repetition_penalty': model_config["repetition_penalty"],
                'no_repeat_ngram_size': model_config["no_repeat_ngram_size"],
                'max_new_tokens': model_config["max_new_tokens"],
                'do_sample': model_config["do_sample"]
            }

            # Model initialization
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ).eval()
            self.streamer = TextStreamer(self.tokenizer)
        else:
            self.tokenizer = None
            self.model = None
            self.streamer = None
            self.gen_params = None

        self.retriever = retriever

    def prompt(self, comments, aspect: str = None):
        """
        Generate a prompt for a batch of comments, optionally for a specific aspect.
        """
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
        if aspect:
            system += f"\n\nСосредоточься на аспекте: {aspect}\nСосредоточься только на негативных отзывах."
        similar_text = ""
        if self.retriever and comments:
            # Use the first comment for retrieval if available
            similar = self.retriever.retrieve(comments[0])
            if similar:
                similar_text = "\n".join(
                    f"- Comment: {sim[0][:100]}...\n  Recommendation: {sim[1]}" for sim in similar
                )
            else:
                similar_text = "Нет похожих отзывов. Сосредоточься на новых отзывах и контексте компании."
        comments_text = "\n".join(f"- {comment}" for comment in comments[:5]) if comments else "Нет отзывов."
        user = f"Отзывы: \n{comments_text}\nПохожие отзывы и рекомендации: \n{similar_text}"
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{fewshot}\n\n{user}\n\n### Рекомендации: [/INST]"

    def generate(self, comments, aspect: str = None):
        """Generate recommendations for a batch of comments."""
        if not self.use_recommender:
            # Return hardcoded recommendations for the aspect
            return "\n".join(self.hardcoded_recommendations.get(aspect, [
                "1. Обобщённая рекомендация 1.",
                "2. Обобщённая рекомендация 2.",
                "3. Обобщённая рекомендация 3.",
                "4. Обобщённая рекомендация 4."
            ]))
        try:
            inp = self.tokenizer(self.prompt(comments, aspect), return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inp,
                    **self.gen_params
                )
            return self.tokenizer.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            return "Не удалось сгенерировать рекомендацию"

    def generate_batch(self, comments, batch_size=8):
        """Legacy method for per-comment recommendations."""
        outputs = []
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            for comment in batch:
                outputs.append(self.generate([comment]))  # Wrap single comment in a list
        return outputs