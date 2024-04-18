"""
Программа: Отрисовка графиков при кластеризации
Версия: 1.0
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score, \
    calinski_harabasz_score
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_optimal_distance(list_clusters: list,
                               list_score: list) -> Tuple[float, int]:
    """
    Поиск наибольшего расстояния для метода локтя.
    
    Parameters
    ----------
    list_clusters: list
        Список с уникальными кластерами.
    list_score: list
        Список со значенями для поиска оптимальной точки на графике.
    Returns
    -------
    Tuple[float, int]
        Кортеж с наибольшим расстоянием и соответствующим значением кластера.
    """
    x1, y1 = list_clusters[0], list_score[0]
    x2, y2 = list_clusters[-1], list_score[-1]
    A, B, C = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

    max_dist = -np.inf
    max_x = -np.inf
    for num, i in enumerate(list_score[1:-1]):
        x_0, y_0 = list_clusters[1:-1][num], list_score[1:-1][num]
        dist = abs(A * x_0 + B * y_0 + C) / np.sqrt(A**2 + B**2)

        if dist > max_dist:
            max_dist = dist
            max_x = x_0
        else:
            continue
    return max_dist, max_x


def elbow_picture(labels_std: list, labels_min: list, labels_max: list,
                  labels_median: list, type_optimal: list, min_size: int,
                  max_size: int) -> None:
    """
    Метод локтя.

    Функция для вывода графика зависимостей стандартной ошибки, медианного,
    минимального и максимального числа объектов от кол-ва кластеров.
    
    Parameters
    ----------
    labels_std: list 
        Список значений std кол-ва объектов для разбиения на кластеры.
    labels_min: list
        Список с min кол-вом объектов для разбиения на кластеры.
    labels_max: list
        Список с max кол-вом объектов для разбиения на кластеры.
    labels_median: list 
        Список с median кол-вом объектов для разбиения на кластеры.
    type_optimal: list
        Cписок для поиска оптимального значения.
    min_size: int
        Минимальное количество кластеров.
    max_size: int
        Максимальное количество кластеров.

    Returns
    -------
    None
        Выводит график зависимостей стандартной ошибки, медианного,
        минимального и максимального числа объектов от кол-ва кластеров.
    """

    _, opt_cluster = calculate_optimal_distance(range(min_size, max_size + 1),
                                                type_optimal)

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_size, max_size + 1),
             labels_std,
             marker='s',
             color='green',
             label='std')
    plt.plot(range(min_size, max_size + 1),
             labels_min,
             marker='s',
             color='grey',
             linestyle='dashed',
             label='min')
    plt.plot(range(min_size, max_size + 1),
             labels_median,
             marker='o',
             color='skyblue',
             linestyle='dashed',
             label='median')
    plt.plot(range(min_size, max_size + 1),
             labels_max,
             marker='o',
             color='grey',
             linestyle='dashed',
             label='max')
    plt.xlabel('Кластер')
    plt.ylabel('Станд.ошибка / Мин.кластер / Median / Макс.кластер')
    plt.axvline(x=opt_cluster,
                color='black',
                label=f'optimal clust= {opt_cluster}',
                linestyle='dashed')
    plt.legend()
    plt.show()


def silhouette_plot(data: pd.DataFrame,
                    labels: pd.Series,
                    metrics: str = 'euclidean',
                    ax: plt.Axes = None) -> None:
    """
    Функция вывода графика силуэтного скора.
    
    Parameters
    ----------
    data: pd.DataFrame 
        Данные для анализа.
    labels: pd.Series 
        Данные с метками кластеров.
    metrics: str
        Метрика для расчета силуэтного скора.
    ax: plt.Axes
        Объект для построения графика.
    
    Returns
    -------
    None
        Вывод графика силуэтного скора.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(7, 5)

    silhouette_vals = silhouette_samples(data, labels, metric=metrics)

    y_ticks = []
    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.barh(range(y_lower, y_upper),
                cluster_silhouette_vals,
                edgecolor='none',
                height=1)
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Получение средней оценки силуэтного скора и построение графика
    avg_score = np.mean(silhouette_vals)
    ax.axvline(avg_score, linestyle='--', linewidth=1, color='red')
    ax.set_xlabel(f'Silhouette  = {round(avg_score,1)}')
    ax.set_ylabel('Метки кластеров')
    ax.set_title('График силуэта для различных кластеров', y=1.02)


def metric_picture(score_list: list,
                   min_size: int,
                   max_size: int,
                   name_metric: str,
                   optimal: bool = True) -> None:
    """
    Функция для вывода графика зависимости силуэтной оценки от кол-ва кластеров.
    
    Parameters
    ----------
    score_list: list 
        Список со значениями метрики.
    min_size: int
        Минимальное кол-во кластеров.
    max_size: int
        Максимальное кол-во кластеров.
    name_metric: str 
        Название метрики.
    optimal: bool 
        Нужно ли делать поиск кол-ва кластеров по методу локтя.
    
    Returns
    -------
    None
        Вывод графика зависимости силуэтной оценки от кол-ва кластеров.
    """
    plt.figure(figsize=(8, 6))

    if optimal:
        _, opt_cluster = calculate_optimal_distance(range(
            min_size, max_size + 1),
                                                    list_score=score_list)
        plt.plot(range(min_size, max_size + 1), score_list, marker='s')
        plt.axvline(x=opt_cluster,
                    color='black',
                    label=f'optimal clust= {opt_cluster}',
                    linestyle='dashed')
        plt.xlabel('$Clusters$')
        plt.ylabel(f'${name_metric}$')
    else:
        plt.plot(range(min_size, max_size + 1), score_list, marker='s')
        plt.xlabel('$Clusters$')
        plt.ylabel(f'${name_metric}$')

    plt.show()


def plot_size(data: pd.DataFrame, labels: pd.Series, ax: plt.Axes) -> None:
    """
    Фунция для вывода графика размера кластеров.
    
    Parameters
    ----------
    data: pd.DataFrame
        Данные для анализа.
    labels: pd.Series
        Данные с метками кластеров.
    ax: plt.Axes
        Объект для построения графика.
    
    Returns
    -------
    None
        Вывод графика размера кластеров.
    """
    data = data.assign(cluster=labels)
    data = pd.DataFrame(data.groupby("cluster").count().iloc[:, 0])
    data.columns = ["value"]
    data = data.reset_index()
    sns.barplot(data=data, y="cluster", x="value", orient="h", ax=ax)
    ax.set_xlabel("Кол-во объектов")
    ax.set_title("Размер кластеров", y=1.02)


def clustering_analysis(
        data: pd.DataFrame,
        data_scale: np.array,
        embedding: np.array,
        model: sklearn.base.ClusterMixin,
        kwargs: dict,
        min_size: int = 2,
        max_size: int = 12,
        type_train: str = None
) -> Tuple[Dict[int, np.array], Dict[int, np.array]]:
    """
    Функция подбора количества кластеров с выводом графиков.

    Parameters
    ----------
    data: pd.DataFrame
        Данные для кластеризации.
    data_scale: np.array
        Скалированные данные.
    embedding: np.array
        Эмбединги.
    model: sklearn.base.ClusterMixin
        Модель кластеризации.
    kwargs: dict
        Параметры алгоритма кластеризации.
    min_size: int
        Минимальное количество кластеров.
    max_size: int
        Максимальное количество кластеров.
    type_train: str
        По каким данным кластеризуем.

    Returns
    -------
    Tuple[Dict[int, np.array], Dict[int, np.array]]
        Словарь с метками кластеров.
    """
    # списки значения sse, ст ошибки, min, median и max кол-ва кластеров
    if model().__class__.__name__ == "KMeans":
        sse = []
    else:
        labels_std = []
        labels_min = []
        labels_max = []
        labels_median = []

    # списки для метрик calinski_harabasz и silhouette
    calinski_harabasz = []
    silhouette_list = []
    dict_clusters = {}

    for clust in range(min_size, max_size + 1):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(18, 5)

        if model().__class__.__name__ == 'IndexFlatL2':
            if type_train == "embedding":
                data_train = embedding
            else:
                data_train = data

            # инициализация индекса Faiss
            data_train = data_train.astype('float32')
            d = data_train.shape[1]
            index = faiss.IndexFlatL2(d)
            # добавление данных в индекс
            index.add(data_train)
            # получение кластеров
            kmeans = faiss.Kmeans(d, clust, **kwargs)
            kmeans.train(data_train)
            # получение принадлежности к кластерам для каждого вектора
            D, I = kmeans.index.search(data_train, 1)
            dict_clusters[clust] = I.reshape(-1)
        else:
            clf = model(n_clusters=clust, **kwargs)
            # обучим модель
            if type_train == "embedding":
                clf.fit(embedding)
            else:
                if model().__class__.__name__ == "KMeans":
                    clf.fit(data_scale)

                else:
                    clf.fit(data)

            dict_clusters[clust] = clf.labels_

        print(clust, 'clusters')
        print('-' * 100)

        if model().__class__.__name__ == "KMeans":
            # sse
            sse.append(clf.inertia_)
        else:
            # вычисляем кол-во уникальных объектов в каждом кластере
            _, counts = np.unique(dict_clusters[clust], return_counts=True)
            # добавление статистики по размерам кластеров
            labels_std.append(np.std(counts))
            labels_min.append(np.min(counts))
            labels_max.append(np.max(counts))
            labels_median.append(np.median(counts))

        # размер кластеров
        plot_size(data, dict_clusters[clust], ax=axes[0])
        # график силуэтного скора
        silhouette_plot(data, dict_clusters[clust], ax=axes[1])

        # добавляем в список calinski_harabasz соотв значения метрик
        calinski_harabasz.append(
            calinski_harabasz_score(data, dict_clusters[clust]))
        # добавляем в список silhouette_list соотв значения метрик
        silhouette_list.append(silhouette_score(data, dict_clusters[clust]))
        plt.show()

    # график зависимостей SSE или ст ошибки, min, median, max по методу локтя
    if model().__class__.__name__ == "KMeans":
        metric_picture(sse, min_size, max_size, name_metric="SSE")
    else:
        elbow_picture(labels_std=labels_std,
                      labels_min=labels_min,
                      labels_max=labels_max,
                      labels_median=labels_median,
                      type_optimal=labels_median,
                      min_size=min_size,
                      max_size=max_size)

    # график изменения calinski_harabasz в зависимости от кол-ва кластеров
    metric_picture(score_list=calinski_harabasz,
                   min_size=min_size,
                   max_size=max_size,
                   name_metric='Calinski harabasz',
                   optimal=False)
    # график изменения silhouette в зависимости от кол-ва кластеров
    metric_picture(score_list=silhouette_list,
                   min_size=min_size,
                   max_size=max_size,
                   name_metric='Silhouette',
                   optimal=False)

    return dict_clusters
