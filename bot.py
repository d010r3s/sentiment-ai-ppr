BOT_TOKEN = '8026318516:AAEzmVQrPKv1n_7QZJgZBHt659avmOJg1AQ' 

import asyncio
import logging
import os

import pandas as pd
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile, 
    FSInputFile
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.enums import ParseMode

import matplotlib.pyplot as plt
import seaborn as sns

from generate_plots import (
    generate_sentiment_distribution_plot, generate_rating_distribution_plot, 
    generate_avg_rating_per_sentiment_plot, generate_aspect_frequency_plot, 
    generate_aspect_sentiment_distribution_plot
)

ASPECT_LABELS = [
    "приложение", "топливо", "карта", "поддержка",
    "интерфейс", "отчет", "эвакуатор", "цена",
    "страховка", "шины_диски"
]


class AnalysisStates(StatesGroup):
    waiting_for_choice = State()
    viewing_overall = State()
    viewing_aspect = State()

# клавиатуры 

main_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Проанализировать отзывы", callback_data="analyze_reviews")]
])

company_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Моя компания", callback_data="my_company")],
    [InlineKeyboardButton(text="Конкурент", callback_data="competitor")],
    [InlineKeyboardButton(text="Выйти", callback_data="exit")]
])

aspect_buttons = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text=aspect.capitalize(), callback_data=f"aspect_{i}")]
    for i, aspect in enumerate(ASPECT_LABELS)
] + [[InlineKeyboardButton(text="⬅️ Вернуться в главное меню", callback_data="back_to_main")]])

sentiment_kb = InlineKeyboardMarkup(inline_keyboard = [
    [InlineKeyboardButton(text="⬅️ Назад к аспектам", callback_data="my_company")]
])

try:
    all_reviews_df = pd.read_excel("sentiment-ai-ppr/datasets/all_reviews.xlsx", 
                                   engine='openpyxl', header=None)
    all_reviews_df.columns = ["author", "date", "text", "rating", "source", "sentiment"]
    all_reviews_df.dropna(subset=["text", "rating", "sentiment"], inplace=True)
    all_reviews_df['rating'] = pd.to_numeric(all_reviews_df['rating'], errors='coerce').fillna(3).astype(int).clip(1, 5)
    labeled_df = pd.read_csv("sentiment-ai-ppr/datasets/labeled_reviews.csv")
    labeled_df = labeled_df[[*ASPECT_LABELS]]
    labeled_df = labeled_df.astype(int)

except Exception as e:
    print(f"[!] Ошибка при загрузке данных: {e}")
    exit(1)

###############################

router = Router()

@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    welcome_text = (
        "Добро пожаловать! Мы - команда SentimentAI.\n\n"
        "Мы тщательно собирали фидбек ваших клиентов и поможем Вам автоматизировать анализ отзывов."
    )
    await message.answer(welcome_text, reply_markup=main_kb)

@router.callback_query(F.data == "analyze_reviews")
async def choose_company(query: CallbackQuery, state: FSMContext):
    await query.message.edit_text("Выберите, чьи отзывы вы хотите проанализировать:", reply_markup=company_kb)
    await state.set_state(AnalysisStates.waiting_for_choice)

@router.callback_query(F.data == "my_company")
async def show_overall(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_overall)
    msg = query.message

    # График 1: Тональность 
    input_file = generate_sentiment_distribution_plot(all_reviews_df)
    input_file = FSInputFile(input_file)
    await msg.answer_photo(input_file, caption="Распределение по тональности")

    # График 2: Рейтинг
    input_file = generate_rating_distribution_plot(all_reviews_df)
    input_file = FSInputFile(input_file)
    await msg.answer_photo(input_file, caption="Распределение по рейтингу (1–5)")

    # График 3: Средний рейтинг по тональности
    input_file = generate_avg_rating_per_sentiment_plot(all_reviews_df)
    input_file = FSInputFile(input_file)
    await msg.answer_photo(input_file, caption="Средний рейтинг по тональности")

    # График 4: Частота аспектов
    input_file = generate_aspect_frequency_plot(ASPECT_LABELS, labeled_df)
    input_file = FSInputFile(input_file)
    await msg.answer_photo(input_file, caption="Частота аспектов")

    # График 5: распределение тональностей по аспектам
    # input_file = generate_aspect_sentiment_distribution_plot(ASPECT_LABELS, labeled_df)
    # input_file = FSInputFile(input_file)
    # await msg.answer_photo(input_file, caption="Частота аспектов", reply_markup=aspect_buttons)
    await msg.answer(text='<b> здесь должен быть график распределения тональностей по аспектам </b>', 
                     parse_mode='html', 
                     reply_markup=aspect_buttons)

@router.callback_query(F.data == "competitor")
async def show_overall(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_overall)
    msg = query.message

    await msg.answer(text='<b> Здесь будет анализ отзывов на Вашего конкурента. We steel working... </b>', 
                     reply_markup=company_kb)

@router.callback_query(F.data.startswith("aspect_"))
async def show_aspect_sentiment(query: CallbackQuery, state: FSMContext):
    await state.set_state(AnalysisStates.viewing_aspect)
    aspect_index = int(query.data.split("_")[1])
    aspect_name = ASPECT_LABELS[aspect_index]
    
    aspect_mask = labeled_df[aspect_name] == 1
    subset = all_reviews_df.loc[labeled_df[aspect_mask].index]

    pos_count = len(subset[subset["sentiment"] == "positive"])
    neg_count = len(subset[subset["sentiment"] == "negative"])

    output_path = f'plots/plot_for_aspect_{aspect_name}.png'
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["positive", "negative"], y=[pos_count, neg_count], palette="Set2")
    plt.title(f"Тональность по аспекту '{aspect_name}'")
    plt.ylabel("Количество")
    plt.xlabel("Тональность")
    plt.savefig(output_path)
    plt.close()

    input_file = FSInputFile(output_path)
    await query.message.delete()
    await query.message.answer_photo(
        input_file,
        caption=f"Тональность по аспекту '{aspect_name}'"
    )
    await query.message.answer(text=f"<b> Выбрать другой аспект: </b>", 
                               parse_mode = 'html', 
                               reply_markup=aspect_buttons)

@router.callback_query(F.data == "back_to_main")
async def back_to_main_menu(query: CallbackQuery, state: FSMContext):
    await query.message.delete()
    await query.message.answer("Выберите, чьи отзывы вы хотите проанализировать:", reply_markup=company_kb)
    await state.set_state(AnalysisStates.waiting_for_choice)

@router.callback_query(F.data == "exit")
async def exit_analysis(query: CallbackQuery, state: FSMContext):
    await query.message.edit_text("Вы вышли из анализа.")
    await state.clear()

async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    print("[+] Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())