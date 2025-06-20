import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'datasets')

import base64

# Подключаем шрифт Onest
font_path = os.path.join(BASE_DIR, 'fonts', 'Onest-Regular.ttf')
with open(font_path, 'rb') as f:
    font_data = f.read()
font_base64 = base64.b64encode(font_data).decode()

st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'Onest';
        src: url(data:font/truetype;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}
    html, body, .block-container, .stButton>button {{
        font-family: 'Onest', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    all_reviews = pd.read_csv(
        os.path.join(DATA_DIR, 'all_reviews.csv'),
        sep=';',
        encoding='cp1251'
    )
    labeled = pd.read_csv(
        os.path.join(DATA_DIR, 'labeled_reviews.csv'),
        sep=',',
        encoding='utf-8'
    )
    labeled.index = all_reviews.index
    labeled = labeled.drop(columns=['text'], errors='ignore')
    return all_reviews, labeled

from bot import SENTIMENT_COLORS, RATING_COLORS, ASPECT_COLORS, ASPECT_LABELS

def sentiment_distribution_fig(df, colors):
    counts = df['sentiment'].value_counts().reindex(list(colors.keys()), fill_value=0)
    fig = go.Figure(go.Bar(
        x=list(counts.index),
        y=counts.values,
        marker=dict(color=[colors[s] for s in counts.index])
    ))
    fig.update_layout(
        title='Распределение по тональности',
        xaxis_title='Тональность',
        yaxis_title='Количество',
        font=dict(family='Onest', size=14)
    )
    return fig


def rating_distribution_fig(df, colors):
    counts = df['rating'].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str),
        y=counts.values,
        marker=dict(color=[colors.get(str(r), '#333333') for r in counts.index])
    ))
    fig.update_layout(
        title='Распределение рейтингов',
        xaxis_title='Рейтинг',
        yaxis_title='Количество',
        font=dict(family='Onest', size=14)
    )
    return fig


def avg_rating_per_sentiment_fig(df, colors):
    avg = df.groupby('sentiment')['rating'].mean().reindex(list(colors.keys()))
    fig = go.Figure(go.Bar(
        x=list(avg.index),
        y=avg.values,
        marker=dict(color=[colors[s] for s in avg.index])
    ))
    fig.update_layout(
        title='Средний рейтинг по тональности',
        xaxis_title='Тональность',
        yaxis_title='Средний рейтинг',
        font=dict(family='Onest', size=14)
    )
    return fig


def aspect_frequency_fig(labeled_df, labels, colors):
    freq = labeled_df.sum().reindex(labels)
    fig = go.Figure(go.Bar(
        x=labels,
        y=freq.values,
        marker=dict(color=[colors[a] for a in labels])
    ))
    fig.update_layout(
        title='Частота упоминания аспектов',
        xaxis_title='Аспект',
        yaxis_title='Количество упоминаний',
        font=dict(family='Onest', size=14)
    )
    return fig


def aspect_sentiment_distribution_fig(df, labeled_df, labels, colors):
    combined = pd.concat([
        df['sentiment'],
        labeled_df[labels]
    ], axis=1)
    melted = combined.reset_index().melt(
        id_vars=['index', 'sentiment'],
        value_vars=labels,
        var_name='aspect',
        value_name='flag'
    )
    filtered = melted[melted['flag'] == 1]
    pivot = filtered.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=labels, columns=list(colors.keys()), fill_value=0)

    fig = go.Figure()
    for sentiment, color in colors.items():
        if sentiment in pivot.columns:
            fig.add_trace(go.Bar(
                x=labels,
                y=pivot[sentiment].values,
                name=sentiment,
                marker=dict(color=color)
            ))
    fig.update_layout(
        barmode='stack',
        title='Тональности по аспектам',
        xaxis_title='Аспект',
        yaxis_title='Количество',
        font=dict(family='Onest', size=14)
    )
    return fig

all_reviews, labeled = load_data()

st.title('Sentiment AI Interactive Dashboard')
st.write(f'Всего отзывов: {len(all_reviews)}')

tabs = st.tabs(['Обзор', 'Аспекты'])

with tabs[0]:
    st.plotly_chart(sentiment_distribution_fig(all_reviews, SENTIMENT_COLORS), use_container_width=True)
    st.plotly_chart(rating_distribution_fig(all_reviews, RATING_COLORS), use_container_width=True)
    st.plotly_chart(avg_rating_per_sentiment_fig(all_reviews, SENTIMENT_COLORS), use_container_width=True)
    st.plotly_chart(aspect_frequency_fig(labeled, ASPECT_LABELS, ASPECT_COLORS), use_container_width=True)
    st.plotly_chart(
        aspect_sentiment_distribution_fig(all_reviews, labeled, ASPECT_LABELS, SENTIMENT_COLORS),
        use_container_width=True
    )

with tabs[1]:
    aspect = st.selectbox('Выберите аспект', ASPECT_LABELS)
    st.plotly_chart(
        aspect_sentiment_distribution_fig(all_reviews, labeled, [aspect], SENTIMENT_COLORS),
        use_container_width=True
    )
