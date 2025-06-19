import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs("plots", exist_ok=True)

def generate_sentiment_distribution_plot(df, output_path="plots/sentiment_distribution.png"):
    """Распределение по тональностям"""
    sentiment_counts = df['sentiment'].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.values, y=sentiment_counts.index, hue=sentiment_counts.index, palette="viridis", dodge=False)
    plt.title("Распределение отзывов по тональности")
    plt.xlabel("Количество")
    plt.ylabel("Тональность")
    plt.legend().remove() 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_rating_distribution_plot(df, output_path="plots/rating_distribution.png"):
    """Распределение по рейтингу (1–5)"""
    rating_counts = df['rating'].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=rating_counts.index.astype(int), y=rating_counts.values, hue=rating_counts.values, palette="plasma", dodge=False)
    plt.title("Распределение отзывов по рейтингу")
    plt.xlabel("Оценка (1–5)")
    plt.ylabel("Количество")
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_avg_rating_per_sentiment_plot(df, output_path="plots/avg_rating_per_sentiment.png"):
    """Средний рейтинг по каждой тональности"""
    avg_ratings = df.groupby('sentiment')['rating'].mean()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=avg_ratings.values, y=avg_ratings.index, hue=avg_ratings.index, palette="coolwarm", dodge=False)
    plt.title("Средняя оценка по тональности")
    plt.xlabel("Средний рейтинг")
    plt.ylabel("Тональность")
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_aspect_frequency_plot(aspect_labels, df, output_path="plots/aspect_frequency.png"):
    """Частота упоминаний аспектов"""
    aspect_counts = df[aspect_labels].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=aspect_counts.values, y=aspect_counts.index, hue=aspect_counts.index, palette="magma", dodge=False)
    plt.title("Частота упоминаний аспектов")
    plt.xlabel("Количество упоминаний")
    plt.ylabel("Аспект")
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_aspect_sentiment_distribution_plot(aspect_name, pos_count, neg_count, output_path="plots/aspect_sentiment_distribution.png"):
    """Заглушка для распределения тональностей по аспектам"""
    labels = ['positive', 'negative']
    values = [pos_count, neg_count]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, hue=values, palette="Set2", dodge=False)
    plt.title(f"Тональность по аспекту '{aspect_name}'")
    plt.ylabel("Количество")
    plt.xlabel("Тональность")
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path