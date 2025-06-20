import asyncio
import logging
import os
import sqlite3
import json
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
from src.models.recommender import Recommender
from telegram_bot.generate_plots import (
    generate_sentiment_distribution_plot,
    generate_rating_distribution_plot,
    generate_avg_rating_per_sentiment_plot,
    generate_aspect_frequency_plot,
    generate_aspect_sentiment_distribution_plot,
    draw_semi_rounded_gradient,
    style_axes,
    hex_to_rgb
)
from src.utils.config import load_config

config = load_config()
BOT_TOKEN = config["bot"]["BOT_TOKEN"]

# Constants
ASPECT_LABELS = [
    "приложение", "топливо", "карта", "поддержка",
    "интерфейс", "отчет", "эвакуатор", "цена",
    "страховка", "шины_диски"
]
ASPECT_COLORS = {
    "приложение": "#62FFD7", "топливо": "#F99400",
    "карта": "#B2DAFF", "поддержка": "#1890CA",
    "интерфейс": "#0173CF", "отчет": "#62FFD7",
    "эвакуатор": "#F99400", "цена": "#B2DAFF",
    "страховка": "#1890CA", "шины_диски": "#0173CF",
}
SENTIMENT_COLORS = {
    "positive": "#62FFD7",
    "negative": "#F99400",
    "neutral": "#B2DAFF",
}
RATING_COLORS = {
    "1": "#62FFD7", "2": "#F99400",
    "3": "#B2DAFF", "4": "#1890CA",
    "5": "#0173CF",
}

# State machine
class AnalysisStates(StatesGroup):
    waiting_for_choice = State()
    viewing_overall = State()
    viewing_aspect = State()

# Keyboards
main_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Проанализировать отзывы", callback_data="analyze_reviews")]
])
company_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Моя компания", callback_data="my_company")],
    [InlineKeyboardButton(text="Конкурент", callback_data="competitor")],
    [InlineKeyboardButton(text="Выйти", callback_data="exit")]
])

def chunk_list(lst, chunk_size=2):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

aspect_button_rows = chunk_list([
    InlineKeyboardButton(text=asp.capitalize(), callback_data=f"aspect_{i}")
    for i, asp in enumerate(ASPECT_LABELS)
], chunk_size=2)

aspect_button_rows.append([InlineKeyboardButton(text="⬅️ Назад в меню", callback_data="back_to_main")])
aspect_buttons = InlineKeyboardMarkup(inline_keyboard=aspect_button_rows)

# Initialize Recommender globally
recommender = Recommender()


def fetch_feedback_for_brand(db_path, brand):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback WHERE brand = ?", conn, params=(brand,))

    # Safely parse aspect as JSON list
    def parse_aspect(x):
        try:
            return json.loads(x) if x else []
        except Exception:
            return []

    df["aspect_list"] = df["aspect"].apply(parse_aspect)

    # Ensure ratings are present and clean
    if "rating" not in df.columns:
        df["rating"] = None

    # Build graph dataframe
    graph_df = df[["rating", "tone", "comment"]].copy()
    graph_df.rename(columns={"tone": "sentiment", "comment": "text"}, inplace=True)
    if "date" in df.columns:
        graph_df["date"] = df["date"]

    # Build aspect binary dataframe for each aspect
    rec_df = pd.DataFrame()
    rec_df["text"] = df["comment"]
    for aspect in ASPECT_LABELS:
        rec_df[aspect] = df["aspect_list"].apply(lambda aspects: int(aspect in aspects))

    # Optional debug checks
    num_negative = df[df["tone"] == "negative"].shape[0]
    num_with_rec = df[df["recommendations"].notnull()].shape[0]
    print(f"DEBUG: {num_negative} negative comments")
    print(f"DEBUG: {num_with_rec} comments with recommendations")

    return graph_df, rec_df

# Router
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
    config = load_config()
    db_path = config["database"]["path"]
    brand = config["processing"]["default_brand"]
    graph_df, rec_df = fetch_feedback_for_brand(db_path, brand)

    # Generate and send plots
    p1 = generate_sentiment_distribution_plot(graph_df, SENTIMENT_COLORS)
    await msg.answer_photo(FSInputFile(p1), caption="Распределение по тональности")
    p2 = generate_rating_distribution_plot(graph_df, RATING_COLORS)
    await msg.answer_photo(FSInputFile(p2), caption="Рейтинг (1–5)")
    p3 = generate_avg_rating_per_sentiment_plot(graph_df, SENTIMENT_COLORS)
    await msg.answer_photo(FSInputFile(p3), caption="Средний рейтинг по тональности")
    p4 = generate_aspect_frequency_plot(rec_df, ASPECT_LABELS, ASPECT_COLORS)
    await msg.answer_photo(FSInputFile(p4), caption="Частота аспектов")
    p5 = generate_aspect_sentiment_distribution_plot(
        graph_df, rec_df, ASPECT_LABELS, SENTIMENT_COLORS
    )
    await msg.answer_photo(
        FSInputFile(p5), caption="Тональности по аспектам", reply_markup=aspect_buttons
    )


@router.callback_query(F.data == "competitor")
async def show_competitor(query: CallbackQuery, state: FSMContext):
    # await state.set_state(AnalysisStates.viewing_overall)
    # msg = query.message
    # config = load_config()
    # db_path = config["database"]["path"]
    # brand = config["processing"]["competitor_brand"]
    # graph_df, rec_df = fetch_feedback_for_brand(db_path, brand)
    #
    # # Generate and send plots
    # p1 = generate_sentiment_distribution_plot(graph_df, SENTIMENT_COLORS)
    # await msg.answer_photo(FSInputFile(p1), caption="Распределение по тональности")
    # p2 = generate_rating_distribution_plot(graph_df, RATING_COLORS)
    # await msg.answer_photo(FSInputFile(p2), caption="Рейтинг (1–5)")
    # p3 = generate_avg_rating_per_sentiment_plot(graph_df, SENTIMENT_COLORS)
    # await msg.answer_photo(FSInputFile(p3), caption="Средний рейтинг по тональности")
    # p4 = generate_aspect_frequency_plot(rec_df, ASPECT_LABELS, ASPECT_COLORS)
    # await msg.answer_photo(FSInputFile(p4), caption="Частота аспектов")
    # p5 = generate_aspect_sentiment_distribution_plot(
    #     graph_df, rec_df, ASPECT_LABELS, SENTIMENT_COLORS
    # )
    # await msg.answer_photo(
    #     FSInputFile(p5), caption="Тональности по аспектам", reply_markup=aspect_buttons
    # )
    await state.set_state(AnalysisStates.viewing_overall)

    config = load_config()
    db_path = config["database"]["path"]
    brand = config["processing"]["competitor_brand"]
    graph_df, rec_df = fetch_feedback_for_brand(db_path, brand)

    if graph_df.empty:
        await query.message.answer(
            f"Нет данных для бренда: {brand}.",
            parse_mode="HTML",
            reply_markup=company_kb
        )
        return

    await query.message.edit_text(f"Показатели конкурента: {brand}")

    # Generate and send plots
    p1 = generate_sentiment_distribution_plot(graph_df, SENTIMENT_COLORS)
    await query.message.answer_photo(FSInputFile(p1), caption="Распределение по тональности (конкурент)")
    p2 = generate_rating_distribution_plot(graph_df, RATING_COLORS)
    await query.message.answer_photo(FSInputFile(p2), caption="Рейтинг конкурента (1–5)")
    p3 = generate_avg_rating_per_sentiment_plot(graph_df, SENTIMENT_COLORS)
    await query.message.answer_photo(FSInputFile(p3), caption="Средний рейтинг по тональности (конкурент)")
    p4 = generate_aspect_frequency_plot(rec_df, ASPECT_LABELS, ASPECT_COLORS)
    await query.message.answer_photo(FSInputFile(p4), caption="Частота аспектов (конкурент)")
    p5 = generate_aspect_sentiment_distribution_plot(
        graph_df, rec_df, ASPECT_LABELS, SENTIMENT_COLORS
    )
    await query.message.answer_photo(
        FSInputFile(p5),
        caption="Тональности по аспектам (конкурент)",
        reply_markup=aspect_buttons
    )

    # Recommendations block
    subset_rec = graph_df[graph_df['sentiment'] == "negative"]
    comments = subset_rec['text'].tolist()[:5]

    if not comments:
        await query.message.answer(
            f"Нет негативных отзывов для бренда «{brand}», рекомендаций нет.",
            parse_mode="HTML",
            reply_markup=aspect_buttons
        )
        return

    recommendations = recommender.generate(comments)
    await query.message.answer(
        f"<b>Рекомендации по негативным отзывам (бренд «{brand}»):</b>\n\n{recommendations}",
        parse_mode="HTML",
        reply_markup=aspect_buttons
    )


@router.callback_query(F.data.startswith("aspect_"))
async def show_aspect_sentiment(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_aspect)
    idx = int(query.data.split("_")[1])
    aspect = ASPECT_LABELS[idx]

    # Fetch fresh data
    config = load_config()
    db_path = config["database"]["path"]
    brand = config["processing"]["default_brand"]
    graph_df, rec_df = fetch_feedback_for_brand(db_path, brand)

    # Filter comments by aspect only for the plot
    mask_aspect = rec_df[aspect] == 1
    subset_plot = graph_df.loc[mask_aspect]
    sentiments = ["positive", "negative", "neutral"]
    counts = [subset_plot["sentiment"].value_counts().get(s, 0) for s in sentiments]

    # Generate sentiment plot
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, len(sentiments), 0.8, max(counts))
    fig.canvas.draw()
    bg = (1, 1, 1)
    for i, s in enumerate(sentiments):
        col = hex_to_rgb(SENTIMENT_COLORS[s])
        draw_semi_rounded_gradient(ax, i, 0.8, counts[i], col, bg)
    positions = [i + 0.4 for i in range(len(sentiments))]
    ax.set_xticks(positions)
    ax.set_xticklabels(sentiments, rotation=45, ha='right', fontsize=10)
    ax.set_title(f"Тональность по аспекту «{aspect}»", pad=10)
    plt.tight_layout()
    out = f"plots/aspect_{aspect}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.close(fig)

    # Send sentiment plot
    await query.message.delete()
    await query.message.answer_photo(FSInputFile(out), caption=f"Тональность: {aspect}")

    # Filter comments by aspect and negative sentiment for recommendations
    subset_rec = subset_plot[subset_plot['sentiment'] == "negative"]
    comments = subset_rec['text'].tolist()[:5]
    if not comments:
        await query.message.answer(
            f"Нет негативных отзывов для аспекта «{aspect}», рекомендаций нет.",
            parse_mode="HTML",
            reply_markup=aspect_buttons
        )
        return

    recommendations = recommender.generate(comments, aspect=aspect)
    await query.message.answer(
        f"<b>Рекомендации по негативным отзывам для аспекта «{aspect}»:</b>\n\n{recommendations}",
        parse_mode="HTML",
        reply_markup=aspect_buttons
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
    dp = Dispatcher()
    dp.include_router(router)
    print("[+] Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())