"""
Программа: Снижение размерности признаков
Версия: 1.0
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def pca_optimal_comp(data: pd.DataFrame) -> None:
    """
    Определяет оптимальное количество компонент для PCA 
    с учетом объясненной дисперсии.

    Parameters:
    ----------
    data: pd.DataFrame
        Данные.

    Returns:
    -------
    None
        График с оптимальным количеством компонент.
    """
    pca = PCA().fit(data)

    plt.figure(figsize=(12, 5))

    x = np.arange(1, len(pca.explained_variance_ratio_) + 1, 1)
    plt.plot(
        x,
        np.cumsum(pca.explained_variance_ratio_),
        # marker='o',
        linestyle='--',
        color='darkcyan')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.text(0.6, 0.85, '90% cut-off threshold', color='red', fontsize=12)
    plt.grid(axis='x')

    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    plt.show()


def pca_embeddings(data: pd.DataFrame, random_state: int,
                   name: str) -> np.ndarray:
    """
    Применяет PCA с описанием 90% данных и визуализирует результаты.

    Parameters:
    ----------
    data: pd.DataFrame
        Данные.
    random_state: int
        random_state для воспроизводимости.
    name: str
        Название графика.

    Returns:
    -------
    np.ndarray
        Эмбеддинги.
    """
    # PCA с описанием 90% данных
    pca = PCA(n_components=0.9, random_state=random_state)
    # применим pca
    X_embedding_pca = pca.fit_transform(data)

    # эмбеддинги на трехмерном графике
    fig = px.scatter_3d(X_embedding_pca,
                        x=0,
                        y=1,
                        z=2,
                        labels={'color': 'species'},
                        title=f'PCA embeddings 3-d ({name})')
    fig.update_traces(marker_size=2)
    fig.show()

    # эмбеддинги на двумерном графике
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=X_embedding_pca[:, 0], y=X_embedding_pca[:, 1], s=50)
    plt.title(f'PCA embeddings 2-d ({name})')
    plt.show()

    return X_embedding_pca


def umap_embeddings(data: pd.DataFrame, random_state: int,
                    name: str) -> np.ndarray:
    """
    Снижает размерность признаков до 3 компонент при помощи UMAP 
    и визуализирует результаты.
    
    Parameters:
    ----------
    data: pd.DataFrame
        Данные.
    random_state: int
        random_state для воспроизводимости.
    name: str
        Название графика.

    Returns:
    -------
    np.ndarray
        Эмбеддинги.
    """
    # снизим размерность признаков до 3 компонент при помощи umap
    umap = UMAP(n_components=3, random_state=random_state)
    X_embedding_umap = umap.fit_transform(data)

    # эмбеддинги на трехмерном графике
    fig = px.scatter_3d(X_embedding_umap,
                        x=0,
                        y=1,
                        z=2,
                        labels={'color': 'species'},
                        title=f'UMAP embeddings 3-d ({name})')
    fig.update_traces(marker_size=2)
    fig.show()

    # эмбеддинги на двумерном графике
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=X_embedding_umap[:, 0], y=X_embedding_umap[:, 1], s=50)
    plt.title(f'UMAP embeddings 2-d ({name})')
    plt.show()

    return X_embedding_umap
