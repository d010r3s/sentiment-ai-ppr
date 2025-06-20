import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import base64
import sqlite3
from ast import literal_eval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import attachment_processor
from src.models.recommender import Recommender

recommender = Recommender()


BASE_DIR = os.path.dirname(__file__)
# DATA_DIR = os.path.join(BASE_DIR, 'datasets')
FONTS_DIR = os.path.join(BASE_DIR, 'fonts')

font_path = os.path.join(FONTS_DIR, 'Onest-Regular.ttf')
if os.path.exists(font_path):
    with open(font_path, 'rb') as f:
        font_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    @font-face {{ font-family: 'Onest'; src: url(data:font/truetype;base64,{font_b64}) format('truetype'); }}
    html, body, .block-container, .stButton>button {{ font-family: 'Onest', sans-serif; }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    # Point directly to root config.yaml, assuming BASE_DIR is streamlit_app/
    config_path = os.path.join(BASE_DIR, "..", "config", "config.yaml")
    from src.utils.config import load_config  # import from config/config.py
    config = load_config(config_path)  # pass path to load_config

    db_path = config["database"]["path"]
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM feedback", conn)

    df.rename(columns={'tone': 'sentiment', 'comment': 'text'}, inplace=True)
    # Parse JSON strings to proper Python lists (for aspect analysis)
    if 'aspect' in df.columns:
        df['parsed_aspects'] = df['aspect'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) and x.strip().startswith("[") else []
        )
    else:
        df['parsed_aspects'] = [[] for _ in range(len(df))]

    return df


from telegram_bot.bot import SENTIMENT_COLORS, RATING_COLORS, ASPECT_COLORS, ASPECT_LABELS


def sentiment_distribution_fig(df, colors):
    counts = df['sentiment'].value_counts().reindex(colors.keys(), fill_value=0)
    fig = go.Figure(go.Bar(x=list(counts.index), y=counts.values,
                            marker_color=[colors[s] for s in counts.index]))
    fig.update_layout(title='Распределение по тональности',
                      xaxis_title='Тональность', yaxis_title='Количество',
                      font=dict(family='Onest', size=14))
    return fig


def rating_distribution_fig(df, colors):
    counts = df['rating'].value_counts().sort_index()
    bar_colors = [colors.get(str(r), '#333333') for r in counts.index]
    fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values,
                            marker_color=bar_colors))
    fig.update_layout(title='Распределение рейтингов',
                      xaxis_title='Рейтинг', yaxis_title='Количество',
                      font=dict(family='Onest', size=14))
    return fig


def avg_rating_per_sentiment_fig(df, colors):
    avg = df.groupby('sentiment')['rating'].mean().reindex(colors.keys())
    fig = go.Figure(go.Bar(x=list(avg.index), y=avg.values,
                            marker_color=[colors[s] for s in avg.index]))
    fig.update_layout(title='Средний рейтинг по тональности',
                      xaxis_title='Тональность', yaxis_title='Средний рейтинг',
                      font=dict(family='Onest', size=14))
    return fig


def aspect_frequency_fig(labeled_df, labels, colors):
    freq = labeled_df.sum().reindex(labels)
    fig = go.Figure(go.Bar(x=labels, y=freq.values,
                            marker_color=[colors[a] for a in labels]))
    fig.update_layout(title='Частота упоминания аспектов',
                      xaxis_title='Аспект', yaxis_title='Количество упоминаний',
                      font=dict(family='Onest', size=14))
    return fig


def aspect_sentiment_distribution_fig(df, labeled_df, labels, colors):
    combined = pd.concat([df['sentiment'], labeled_df[labels]], axis=1)
    melted = combined.reset_index().melt(id_vars=['index','sentiment'],
                                         value_vars=labels,
                                         var_name='aspect', value_name='flag')
    filtered = melted[melted['flag']==1]
    pivot = filtered.groupby(['aspect','sentiment']).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=labels, columns=colors.keys(), fill_value=0)
    fig = go.Figure()
    for sentiment, color in colors.items():
        if sentiment in pivot.columns:
            fig.add_trace(go.Bar(x=labels, y=pivot[sentiment].values,
                                  name=sentiment, marker_color=color))
    fig.update_layout(barmode='stack', title='Тональности по аспектам',
                      xaxis_title='Аспект', yaxis_title='Количество',
                      font=dict(family='Onest', size=14))
    return fig


df1 = load_data()

# Generate labeled matrix (like labeled_reviews.csv)
labeled = pd.DataFrame(0, index=df1.index, columns=ASPECT_LABELS)
for i, aspect_list in enumerate(df1['parsed_aspects']):
    for a in aspect_list:
        if a in ASPECT_LABELS:
            labeled.at[i, a] = 1

# Now brand filter UI
brands = df1['brand'].unique().tolist()
selected_brand = st.selectbox("Бренд", brands)
df = df1[df1['brand'] == selected_brand]
labeled = labeled.loc[df.index]

st.title('Sentiment AI Dashboard')
st.write(f'Исходных отзывов: {len(df1)}')

uploaded = st.file_uploader("Загрузите дополнительный CSV с отзывами. Обязательные колонки: text, sentiment, rating", type=['csv'],
                            help='CSV must contain text and sentiment columns')
if uploaded:
    try:
        # Pass raw bytes to your processor
        file_bytes = uploaded.read()
        user_df = attachment_processor.process_attachment(file_bytes)

        combined = pd.concat([df1, user_df], ignore_index=True)
        if 'id' in combined.columns:
            combined = combined.drop_duplicates(subset='id', keep='first')
        df1 = combined

        # regenerate labeled matrix
        labeled = pd.DataFrame(0, index=df1.index, columns=ASPECT_LABELS)
        for i, aspect_list in enumerate(df1['parsed_aspects']):
            for a in aspect_list:
                if a in ASPECT_LABELS:
                    labeled.at[i, a] = 1

        # reapply brand filter
        df = df1[df1['brand'] == selected_brand]
        labeled = labeled.loc[df.index]

        st.success(f'Добавлено {len(user_df)} отзывов, всего теперь {len(df1)}')
    except Exception as e:
        st.error(f'Ошибка при загрузке файла: {e}')


# Общий вывод
st.write(f'Всего отзывов для анализа: {len(df)}. Дубликаты удалены.')
tabs = st.tabs(['Обзор','Аспекты','Рекомендации'])
with tabs[0]:
    st.plotly_chart(sentiment_distribution_fig(df, SENTIMENT_COLORS), use_container_width=True)
    st.plotly_chart(rating_distribution_fig(df, RATING_COLORS), use_container_width=True)
    st.plotly_chart(avg_rating_per_sentiment_fig(df, SENTIMENT_COLORS), use_container_width=True)
    st.plotly_chart(aspect_frequency_fig(labeled, ASPECT_LABELS, ASPECT_COLORS), use_container_width=True)
    st.plotly_chart(aspect_sentiment_distribution_fig(df, labeled, ASPECT_LABELS, SENTIMENT_COLORS), use_container_width=True)
with tabs[1]:
    aspect = st.selectbox('Выберите аспект', ASPECT_LABELS)
    st.plotly_chart(aspect_sentiment_distribution_fig(df, labeled, [aspect], SENTIMENT_COLORS), use_container_width=True)
with tabs[2]:
    st.header('Рекомендации')
    aspect = st.selectbox('Выберите аспект для рекомендаций', ASPECT_LABELS)

    # Filter negative comments for the selected aspect
    subset_rec = df[(df['sentiment'] == 'negative') & (labeled[aspect] == 1)]
    comments = subset_rec['text'].tolist()[:5]

    if comments:
        recommendations = recommender.generate(comments, aspect=aspect)
        st.write(recommendations)
    else:
        st.info(f"Нет негативных отзывов для аспекта «{aspect}», рекомендаций нет.")
