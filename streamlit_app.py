import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import base64
import io
import csv

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
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
    reviews_path = os.path.join(DATA_DIR, 'all_reviews.csv')
    if os.path.exists(reviews_path):
        reviews = pd.read_csv(reviews_path, sep=';', encoding='cp1251')
    else:
        reviews = pd.DataFrame(columns=['id', 'text', 'rating', 'sentiment'])
    labels_path = os.path.join(DATA_DIR, 'labeled_reviews.csv')
    if os.path.exists(labels_path):
        labeled = pd.read_csv(labels_path, sep=',', encoding='utf-8')
        if 'text' in labeled.columns:
            labeled.index = reviews.index
            labeled = labeled.drop(columns=['text'], errors='ignore')
    else:
        labeled = pd.DataFrame()
    return reviews, labeled

from bot import SENTIMENT_COLORS, RATING_COLORS, ASPECT_COLORS, ASPECT_LABELS

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

df_base, labeled = load_data()

st.title('Sentiment AI Dashboard')
st.write(f'Исходных отзывов: {len(df_base)}')

df = df_base.copy()
uploaded = st.file_uploader('Загрузите дополнительный CSV с отзывами', type=['csv'],
                            help='id;text;rating;sentiment')
if uploaded:
    raw = uploaded.getvalue()
    text = None
    for enc in ('cp1251','utf-8','latin1'):
        try:
            text = raw.decode(enc)
            break
        except:
            continue
    if text:
        try:
            dialect = csv.Sniffer().sniff(text[:1024], delimiters=';,\t')
            sep = dialect.delimiter
        except:
            sep = ';'
        user_df = pd.read_csv(io.StringIO(text), sep=sep)
        combined = pd.concat([df_base, user_df], ignore_index=True)
        if 'id' in combined.columns:
            combined = combined.drop_duplicates(subset='id', keep='first')
        added = len(combined) - len(df_base)
        df = combined
        if 'id' in df.columns:
            df = df.drop_duplicates(subset='id', keep='first')
        st.success(f'Добавлено {len(user_df)} отзывов, всего теперь {len(df)}')

# Общий вывод
st.write(f'Всего отзывов для анализа: {len(df)}')
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
    st.info('placeholder')
