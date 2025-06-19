BOT_TOKEN = 'BOT_TOKEN'

import asyncio
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from generate_plots import (
    generate_sentiment_distribution_plot,
    generate_rating_distribution_plot,
    generate_avg_rating_per_sentiment_plot,
    generate_aspect_frequency_plot,
    generate_aspect_sentiment_distribution_plot,
    draw_semi_rounded_gradient,
    style_axes,
    hex_to_rgb
)

# === Константы: аспекты и палитры ===
ASPECT_LABELS = [
    "приложение", "топливо", "карта", "поддержка",
    "интерфейс", "отчет", "эвакуатор", "цена",
    "страховка", "шины_диски"
]
ASPECT_COLORS = {
    "приложение": "#62FFD7", "топливо": "#F99400",
    "карта": "#B2DAFF",     "поддержка": "#1890CA",
    "интерфейс": "#0173CF", "отчет": "#62FFD7",
    "эвакуатор": "#F99400", "цена": "#B2DAFF",
    "страховка": "#1890CA","шины_диски": "#0173CF",
}
SENTIMENT_COLORS = {
    "positive": "#62FFD7",
    "negative": "#F99400",
    "neutral":  "#B2DAFF",
}
RATING_COLORS = {
    "1": "#62FFD7", "2": "#F99400",
    "3": "#B2DAFF", "4": "#1890CA",
    "5": "#0173CF",
}

class AnalysisStates(StatesGroup):
    waiting_for_choice = State()
    viewing_overall    = State()
    viewing_aspect     = State()

# === Клавиатуры ===
main_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Проанализировать отзывы", callback_data="analyze_reviews")]
])
company_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Моя компания", callback_data="my_company")],
    [InlineKeyboardButton(text="Конкурент",   callback_data="competitor")],
    [InlineKeyboardButton(text="Выйти",       callback_data="exit")]
])
aspect_buttons = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text=asp.capitalize(), callback_data=f"aspect_{i}")]
    for i, asp in enumerate(ASPECT_LABELS)
] + [[InlineKeyboardButton(text="⬅️ Назад в меню", callback_data="back_to_main")]])
sentiment_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="⬅️ Назад к аспектам", callback_data="my_company")]
])

# === Загрузка данных ===
try:
    all_reviews_df = pd.read_excel(
        "datasets/all_reviews.xlsx",
        engine="openpyxl", header=None
    )
    all_reviews_df.columns = ["author","date","text","rating","source","sentiment"]
    all_reviews_df.dropna(subset=["text","rating","sentiment"], inplace=True)
    all_reviews_df["rating"] = (
        pd.to_numeric(all_reviews_df["rating"], errors="coerce")
          .fillna(3).astype(int).clip(1,5)
    )

    labeled_reviews = pd.read_csv(
        "datasets/labeled_reviews.csv", sep=",", encoding="utf-8"
    )
    labeled_df = labeled_reviews[ASPECT_LABELS].astype(int)

except Exception as e:
    print(f"[!] Ошибка при загрузке данных: {e}")
    exit(1)

# === Настройка роутера ===
router = Router()

@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Добро пожаловать! Мы — SentimentAI.\n\n"
        "Автоматизируем анализ отзывов ваших клиентов.",
        reply_markup=main_kb
    )

@router.callback_query(F.data == "analyze_reviews")
async def choose_company(query: CallbackQuery, state: FSMContext):
    await query.message.edit_text("Кого анализировать?", reply_markup=company_kb)
    await state.set_state(AnalysisStates.waiting_for_choice)

@router.callback_query(F.data == "my_company")
async def show_overall(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_overall)
    msg = query.message

    # 1) Тональность
    p1 = generate_sentiment_distribution_plot(all_reviews_df, SENTIMENT_COLORS)
    await msg.answer_photo(FSInputFile(p1), caption="Распределение по тональности")

    # 2) Рейтинг
    p2 = generate_rating_distribution_plot(all_reviews_df, RATING_COLORS)
    await msg.answer_photo(FSInputFile(p2), caption="Рейтинг (1–5)")

    # 3) Средний рейтинг по тональности
    p3 = generate_avg_rating_per_sentiment_plot(all_reviews_df, SENTIMENT_COLORS)
    await msg.answer_photo(FSInputFile(p3), caption="Средний рейтинг по тональности")

    # 4) Частота аспектов
    p4 = generate_aspect_frequency_plot(labeled_df, ASPECT_LABELS, ASPECT_COLORS)
    await msg.answer_photo(FSInputFile(p4), caption="Частота аспектов")

    # 5) Распределение тональностей по аспектам
    p5 = generate_aspect_sentiment_distribution_plot(
        all_reviews_df,
        labeled_df,
        ASPECT_LABELS,
        SENTIMENT_COLORS
    )
    await msg.answer_photo(
        FSInputFile(p5),
        caption="Тональности по аспектам",
        reply_markup=aspect_buttons
    )

@router.callback_query(F.data == "competitor")
async def show_competitor(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_overall)
    await query.message.answer(
        "<b>Анализ конкурента ещё в разработке...</b>",
        parse_mode="HTML", reply_markup=company_kb
    )

@router.callback_query(F.data.startswith("aspect_"))
async def show_aspect_sentiment(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_aspect)
    idx    = int(query.data.split("_")[1])
    aspect = ASPECT_LABELS[idx]

    # Отфильтровать отзывы по выбранному аспекту
    mask   = labeled_df[aspect] == 1
    subset = all_reviews_df.loc[labeled_df[mask].index]

    sentiments = ["positive", "negative", "neutral"]
    counts     = [(subset["sentiment"] == s).sum() for s in sentiments]

    # Собственный градиентный график
    fig, ax = plt.subplots(figsize=(8,5))
    style_axes(ax, len(sentiments), 0.8, max(counts))
    fig.canvas.draw()

    bg = (1,1,1)
    for i, s in enumerate(sentiments):
        col = hex_to_rgb(SENTIMENT_COLORS[s])
        draw_semi_rounded_gradient(ax, i, 0.8, counts[i], col, bg)

    # Подписи под столбцами
    positions = [i + 0.4 for i in range(len(sentiments))]
    ax.set_xticks(positions)
    ax.set_xticklabels([s for s in sentiments],
                       rotation=45, ha='right', fontsize=10)

    ax.set_title(f"Тональность по аспекту «{aspect}»", pad=10)
    plt.tight_layout()

    out = f"plots/aspect_{aspect}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close(fig)

    await query.message.delete()
    await query.message.answer_photo(FSInputFile(out), caption=f"Тональность: {aspect}")
    await query.message.answer(
        "<b>⬅️ Назад</b>", parse_mode="HTML", reply_markup=aspect_buttons
    )

@router.callback_query(F.data == "back_to_main")
async def back_to_main(query: CallbackQuery, state: FSMContext):
    await query.message.delete()
    await query.message.answer("Выберите категорию:", reply_markup=company_kb)
    await state.set_state(AnalysisStates.waiting_for_choice)

@router.callback_query(F.data == "exit")
async def exit_analysis(query: CallbackQuery, state: FSMContext):
    await query.message.edit_text("Вы вышли из анализа.")
    await state.clear()

async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=BOT_TOKEN)
    dp  = Dispatcher()
    dp.include_router(router)
    print("[+] Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
