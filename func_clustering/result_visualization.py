"""
Программа: Визуализация и интерпретация кластеров
Версия: 1.0
"""

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def clusters_visual(clusters: np.ndarray, X_embedding_pca: np.ndarray,
                    X_embedding_umap: np.ndarray, name: str) -> None:
    """
    Визуализирует кластеры данных в двухмерном и трехмерном пространстве 
    для PCA и UMAP.

    Parameters
    ----------
    clusters: np.ndarray
        Предсказанные значения кластеров.
    X_embedding_pca: np.ndarray
        Эмбеддинги PCA.
    X_embedding_umap: np.ndarray
        Эмбеддинги UMAP.
    name: str
        Название графика.

    Returns
    ------- 
    None
        Трехмерные и двумерные графики с визуализацией кластеров.
    """
    # 3d графики
    # создаем общий subplot для двух графиков
    fig = make_subplots(rows=1,
                        cols=2,
                        specs=[[{
                            'type': 'scatter3d'
                        }, {
                            'type': 'scatter3d'
                        }]],
                        subplot_titles=(f'PCA 3-d ({name})',
                                        f'UMAP 3-d ({name})'))

    # добавляем график PCA
    fig.add_trace(go.Scatter3d(x=X_embedding_pca[:, 0],
                               y=X_embedding_pca[:, 1],
                               z=X_embedding_pca[:, 2],
                               mode='markers',
                               marker=dict(size=2, color=clusters),
                               name='PCA'),
                  row=1,
                  col=1)

    # добавляем график UMAP
    fig.add_trace(go.Scatter3d(x=X_embedding_umap[:, 0],
                               y=X_embedding_umap[:, 1],
                               z=X_embedding_umap[:, 2],
                               mode='markers',
                               marker=dict(size=2, color=clusters),
                               name='UMAP'),
                  row=1,
                  col=2)

    fig.update_traces(marker_size=2)

    fig.show()

    # 2d графики
    # создаем общий subplot для двух графиков
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    # добавляем график PCA
    sns.scatterplot(x=X_embedding_pca[:, 0],
                    y=X_embedding_pca[:, 1],
                    c=clusters,
                    ax=axes[0])
    axes[0].set_title(f'PCA 2-d ({name})')

    # добавляем график UMAP
    sns.scatterplot(x=X_embedding_umap[:, 0],
                    y=X_embedding_umap[:, 1],
                    c=clusters,
                    ax=axes[1])
    axes[1].set_title(f'UMAP 2-d ({name})')

    plt.show()


def plot_object_features(data: pd.DataFrame, labels: np.ndarray,
                         object_cols: List[str]) -> None:
    """
    Функция для построения распределения категориальных признаков по кластерам.

    Parameters
    ----------
    data: pd.DataFrame
        Входной DataFrame с данными.
    labels: np.ndarray
        Массив меток кластеров.
    object_cols: List[str]
        Список названий колонок категориальных признаков.
        
    Returns
    ------- 
    None
        Выводит график barplot.
    """
    data_label = data.assign(cluster=labels)
    rows = len(data_label['cluster'].unique())
    cols = len(object_cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows))
    fig.tight_layout(pad=5.0)

    for i, cluster_label in enumerate(data_label['cluster'].unique()):
        cluster_data = data_label[data_label['cluster'] == cluster_label]

        for j, feature in enumerate(object_cols):
            feature_counts = cluster_data[feature].value_counts(
                normalize=True).head(5)

            ax = axes[i, j]
            feature_counts.plot(kind='bar', alpha=0.7, ax=ax)
            ax.set_title(f'кластер {cluster_label}')
            ax.set_ylabel('Проценты')
            shortened_labels = [
                label[:30] + '...' if len(str(label)) > 10 else label
                for label in feature_counts.index
            ]
            ax.set_xticklabels(shortened_labels)

    plt.tight_layout()
    plt.show()


def plot_num_features(data: pd.DataFrame, labels: np.array,
                      num_cols: List[str]) -> None:
    """
    График kdeplot для сравнения признаков типа int/float между кластерами.
    
    Parameters
    ----------
    data: pd.DataFrame
        Данные для анализа.
    labels: np.array
        Метки кластеров.
    object_cols: List[str]
        Список с колонками типа int/float.

    Returns
    ------- 
    None
        Выводит график kdeplot.
    """

    n_cols = len(num_cols)
    rows = n_cols // 3 + n_cols % 3
    data_label = data.assign(cluster=labels)

    _, axes = plt.subplots(ncols=3, nrows=rows, figsize=(20, 5 * rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for num, col in enumerate(num_cols):
        sns.boxplot(
            data=data_label,
            y=col,
            x="cluster",
            palette="crest",
            ax=axes.reshape(-1)[num],
        )


def plot_text_features(data: pd.DataFrame, labels: np.ndarray,
                       tfidf_features: List[str]) -> None:
    """
    Функция для построения средних значений текстовых признаков по кластерам.

    Parameters
    ----------
    data: pd.DataFrame
        Входной DataFrame с данными.
    labels: np.ndarray
        Массив меток кластеров.
    tfidf_features: List[str]
        Список названий колонок TF-IDF.

    Returns
    ------- 
    None
        Выводит график barplot.    
    """
    data_label = data.assign(cluster=labels)
    n_cols = 3
    rows = len(data_label['cluster'].unique()) // n_cols + 1

    _, axes = plt.subplots(ncols=3, nrows=rows, figsize=(20, 4 * rows))

    for num, cluster_label in enumerate(data_label['cluster'].unique()):
        cluster_data = data_label[data_label['cluster'] == cluster_label]
        cluster_text = cluster_data[tfidf_features].sum().sort_values(
            ascending=False)
        top_words = cluster_text.head(7).index.tolist()
        ax = axes.reshape(-1)[num]
        ax.bar(
            top_words,
            cluster_text.head(7).values,
            alpha=0.7,
            label=f'Кластер {cluster_label}',
        )
        ax.set_title(f'Top-5 слов TF-IDF для кластера {cluster_label}')
        ax.set_xlabel('Слова')
        ax.set_ylabel('Суммарное значение TF-IDF')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    plt.tight_layout()
    plt.show()
