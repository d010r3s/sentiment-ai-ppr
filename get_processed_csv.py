import pandas as pd
import re

ASPECT_KEYWORDS = {
    "приложение": r"(приложени[ея]|app|программа|мобильное приложение)",
    "топливо": r"(топливо|бензин|дизель|АЗС|заправк[аи])",
    "карта": r"(карта|карт[ау]|выдача карты|карточка|топливная карта)",
    "поддержка": r"(поддержка|служба поддержки|менеджер|сотрудник|консультант|звонок|менеджеру|Азе|Олеся|Тарасенко)",
    "интерфейс": r"(интерфейс|кабинет|личный кабинет|платформа|система|ЛК|locator)",
    "администрация": r"(менеджер|менеджеры|менеджера|сервис|обслуживание|обращение|общение)",
    "цена": r"(цена|стоимость|дешевле|дороже|адекватные цены|завышена|доступная)",
    "страховка": r"(страховк[аи]|страхование|ОСАГО|пыолис)",
    "шины_диски": r"(шин[ау]|диск[ау]|резина|поставка шин|летняя резина)"
}


def extract_aspects(text):
    found = {}
    text_lower = text.lower()
    for aspect, pattern in ASPECT_KEYWORDS.items():
        match = re.search(pattern, text_lower)
        found[aspect] = 1 if match else 0
    return found


# загрузка данных из Excel или CSV
file_path = 'sentiment-ai-ppr/datasets/all_reviews.xlsx'
print(f"[+] Чтение файла: {file_path}...")

try:
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    else:
        raise ValueError("Неподдерживаемый формат файла")
except Exception as e:
    print(f"[!] Ошибка при чтении файла: {e}")
    exit()


if 'text' in df.columns:
    texts = df['text'].dropna().astype(str).tolist()
    print(f"[+] Найдено {len(texts)} комментариев в колонке 'text'")
else:
    print("[!] Колонка 'text' не найдена в файле")
    exit()


print("[+] Разметка аспектов...")

labeled_data = []

for text in texts:
    aspects = extract_aspects(str(text))
    labeled_data.append({
        "text": text,
        **aspects
    })

# сохранение файла
result_df = pd.DataFrame(labeled_data)
result_df.to_csv("sentiment-ai-ppr/datasets/labeled_reviews.csv", index=False)
print(f"Размеченные данные сохранены в labeled_reviews.csv")