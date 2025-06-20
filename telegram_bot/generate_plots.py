import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import font_manager

matplotlib.use('Agg')

font_path = "fonts/Onest-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = 'Onest'
else:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
# Устанавливаем шрифты по-умолчанию на Onest
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Onest']


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255. for i in (0, 2, 4))


def draw_semi_rounded_gradient(ax, x, width, height, color, background=(1,1,1), n=200, arc_pts=100):
    trans = ax.transData
    px_x = trans.transform((x + width/2 + 1, 0))[0] - trans.transform((x + width/2, 0))[0]
    px_y = trans.transform((0, height))[1] - trans.transform((0, height - 1))[1]
    r_disp = (width / 2) * px_x
    r_x = width / 2
    r_y = r_disp / px_y
    cx, cy = x + r_x, height - r_y

    verts = [(x, 0), (x + width, 0), (x + width, height - r_y)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
    thetas = np.linspace(0, np.pi, arc_pts)
    for theta in thetas:
        verts.append((cx + r_x * np.cos(theta), cy + r_y * np.sin(theta)))
        codes.append(Path.LINETO)
    verts.append((x, height - r_y))
    codes.append(Path.LINETO)
    verts.append((x, 0))
    codes.append(Path.CLOSEPOLY)

    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='none', transform=ax.transData)
    ax.add_patch(patch)

    grad = np.linspace(0, 1, n)[:, None]
    grad_colors = np.zeros((n, 1, 4))
    for i, t in enumerate(grad[:, 0]):
        c = [background[j] * (1 - t) + color[j] * t for j in range(3)]
        grad_colors[i, 0, :3] = c
        grad_colors[i, 0, 3] = 1.0
    im = ax.imshow(grad_colors, aspect='auto', extent=(x, x + width, 0, height), origin='lower', zorder=1)
    im.set_clip_path(patch)


def style_axes(ax, n_bars, bar_width, max_h):
    ax.set_xlim(-0.5, n_bars - 0.5 + bar_width)
    ax.set_ylim(0, max_h * 1.1)
    for sp in ['top', 'left']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'right']:
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color('#333333')
        ax.spines[sp].set_linewidth(1)


def generate_sentiment_distribution_plot(df, colors, output_path="plots/sentiment_distribution.png"):
    df = df[df['sentiment'].isin(colors.keys())]
    counts = df['sentiment'].value_counts()
    labels = counts.index.tolist()
    values = counts.values
    fig, ax = plt.subplots(figsize=(8,5))

    if len(values) == 0 or values.max() == 0:
        max_val = 1  # fallback so max always valid
    else:
        max_val = values.max()

    style_axes(ax, len(values), 0.8, values.max())
    fig.canvas.draw()
    bg = (1,1,1)
    for i, lab in enumerate(labels):
        draw_semi_rounded_gradient(ax, i, 0.8, values[i], hex_to_rgb(colors.get(lab, '#888888')), bg)
    positions = [i + 0.4 for i in range(len(labels))]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_title("Распределение отзывов по тональности", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_rating_distribution_plot(df, colors, output_path="plots/rating_distribution.png"):
    counts = df['rating'].value_counts().sort_index()
    labels = counts.index.astype(str).tolist()
    values = counts.values
    fig, ax = plt.subplots(figsize=(8,5))
    style_axes(ax, len(values), 0.8, values.max())
    fig.canvas.draw()
    bg = (1,1,1)
    for i, lab in enumerate(labels):
        draw_semi_rounded_gradient(ax, i, 0.8, values[i], hex_to_rgb(colors.get(lab, '#888888')), bg)
    positions = [i + 0.4 for i in range(len(labels))]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
    ax.set_title("Распределение отзывов по рейтингу", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_avg_rating_per_sentiment_plot(df, colors, output_path="plots/avg_rating_per_sentiment.png"):
    df_filtered = df[df['sentiment'].isin(colors.keys())]
    grouped = df_filtered.groupby('sentiment')['rating'].mean()
    labels = grouped.index.tolist()
    values = grouped.values
    fig, ax = plt.subplots(figsize=(8,5))
    style_axes(ax, len(labels), 0.8, max(values))
    fig.canvas.draw()
    bg = (1,1,1)
    for i, lab in enumerate(labels):
        col = hex_to_rgb(colors.get(lab, '#888888'))
        draw_semi_rounded_gradient(ax, i, 0.8, values[i], col, bg)
    # подписи категорий
    positions = [i + 0.4 for i in range(len(labels))]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_title("Средний рейтинг по тональности", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_aspect_frequency_plot(df, aspects, colors, output_path="plots/aspect_frequency.png"):
    counts = df[aspects].sum()
    labels = aspects
    values = counts.values
    fig, ax = plt.subplots(figsize=(8,5))
    style_axes(ax, len(values), 0.8, values.max())
    fig.canvas.draw()
    bg = (1,1,1)
    for i, lab in enumerate(labels):
        draw_semi_rounded_gradient(ax, i, 0.8, values[i], hex_to_rgb(colors.get(lab, '#888888')), bg)
    positions = [i + 0.4 for i in range(len(labels))]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_title("Частота упоминаний аспектов", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_aspect_sentiment_distribution_plot(reviews_df, aspects_df, aspects, colors, output_path="plots/aspect_sentiment_distribution.png"):
    sentiments = reviews_df['sentiment'].unique().tolist()
    n_groups = len(aspects)
    n_sents = len(sentiments)
    bar_w = 0.8 / n_sents
    max_count = 0
    for asp in aspects:
        mask = aspects_df[asp] == 1
        cnts = reviews_df.loc[aspects_df[mask].index, 'sentiment'].value_counts()
        max_count = max(max_count, cnts.max() if not cnts.empty else 0)
    fig, ax = plt.subplots(figsize=(max(8, n_groups*1.5), 5))
    style_axes(ax, n_groups * n_sents, bar_w, max_count)
    fig.canvas.draw()
    bg = (1,1,1)
    for i, asp in enumerate(aspects):
        mask = aspects_df[asp] == 1
        idxs = aspects_df[mask].index
        for j, sen in enumerate(sentiments):
            cnt = reviews_df.loc[idxs, 'sentiment'].value_counts().get(sen, 0)
            draw_semi_rounded_gradient(ax, i*n_sents + j, bar_w, cnt, hex_to_rgb(colors.get(sen, '#888888')), bg)
    # подписи групп аспектов
    tick_positions = [i * n_sents + (n_sents*bar_w)/2 for i in range(n_groups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(aspects, rotation=45, ha='right', fontsize=10)
    ax.set_title("Распределение тональностей по аспектам", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    return output_path
