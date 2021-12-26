import matplotlib.pyplot as plt
import numpy as np
from seaborn import ecdfplot, distplot
from scipy.stats import chi2, norm
import pandas as pd


def sample_average(x):
    """
    возвращает выборочное среднее
    """
    ret = np.mean(x)
    return ret


def sample_variance(x, x_avg):
    """
    возвращает выборочную дисперсию
    """
    ret = sum((x - x_avg) ** 2) / (len(x))
    return ret


def unb_sample_variance(x, x_avg):
    """
    возвращает несмещённую выборочную дисперсию
    """
    ret = sum((x - x_avg) ** 2) / (len(x) - 1)
    return ret


def min_max_ord_stats(x):
    """
    возвращает (минимальная, максимальная) порядковые статистики
    """
    x = x.sort_values()
    return x.values[0], x.values[len(x.values) - 1]


def delta_val(x_min, x_max):
    """
    возвращает размах
    """
    ret = x_max - x_min
    return ret


def median(x):
    """
    возвращает медиану
    """
    if len(x) % 2 == 0:
        return x.values[int((len(x) + 1) / 2)]
    else:
        return (x.values[int(len(x) / 2)] + x.values[int((len(x) / 2) + 1)]) / 2


def empirical(x):
    """
    график эмпирической функции распределения
    """
    plt.title("№2.2. график эмпирической функции распределения. \nЗакройте окно для продолжения")
    ecdfplot(x)
    plt.show()


def histogram(x):
    """
    график гистограммы относительных частот
    """
    plt.title("№2.2. график гистограммы относительных частот. \nЗакройте окно для продолжения")
    plt.hist(x, histtype='bar', bins=7)
    plt.show()


def kernel_estimation(x):
    """
    ядерная оценка
    """
    plt.title("№2.2. график ядерной оценки. \nЗакройте окно для продолжения")
    plt.xlabel('Ширина лепестка')
    distplot(x, hist=True)
    plt.show()


def is_normal_distribution(x, x_avg, x_var):
    """
    проверяет распределение

    """
    d_plus = 0
    for i, x_1 in enumerate(x):
        d_plus = max(d_plus, (i + 1) / len(x) - norm(loc=x_avg, scale=x_var).cdf(x_1))

    d_minus = 0
    for i, x_1 in enumerate(x):
        d_minus = max(d_minus, norm(loc=x_avg, scale=x_var).cdf(x_1) - i / len(x))
    if (6 * len(x) * max(d_minus, d_plus) + 1) / (6 * np.sqrt(len(x))) <= 1.0599:
        return "Да"
    else:
        return "Нет"


def main():
    df = pd.read_csv("iris.data", delimiter=',')
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = df[df['class'].isin(['Iris-virginica'])]
    petal_w = df['petal_width']
    print("№2.1")
    avg = sample_average(petal_w)
    print("выборочное среднее = ", avg)
    var = sample_variance(petal_w, avg)
    print("выборочная дисперсия = ", var)
    print("несмещенная выборочная дисперсия = ", unb_sample_variance(petal_w, avg))
    min_stat, max_stat = min_max_ord_stats(petal_w)
    print("минимальная порядковая статистика:", min_stat, "\nмаксимальную минимальная порядковая статистика: ",
          max_stat)
    print("размах = ", delta_val(min_stat, max_stat))
    print("медиана = ", median(petal_w))

    empirical(petal_w)
    histogram(petal_w)
    kernel_estimation(petal_w)
    print("№2.4.\nЯвляется ли нормальным распределение?",
          is_normal_distribution(petal_w, avg, np.sqrt(var)))


if __name__ == '__main__':
    main()
