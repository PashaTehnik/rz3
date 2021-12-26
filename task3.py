import pandas as pd
import numpy as np


def task1(petal_l, petal_w):
    petal_l = pd.qcut(petal_l, 4)
    petal_w = pd.qcut(petal_w, 4)

    df = pd.concat([petal_l, petal_w], axis=1)
    table = pd.crosstab(df['petal_length'], df['petal_width'])

    table['vit'] = table.sum(axis=1)
    vtj = pd.Series(name='vtj', dtype='float64')
    table = table.append(vtj, ignore_index=False)
    table.iloc[4] = table.sum(axis=0)

    s = 0
    for i in range(4):
        for j in range(4):
            s += np.power(table.iloc[j][i], 2) / (table.iloc[4][i] * table.iloc[j][4])
    hi_prac = (s - 1) * 50

    hi_teor = 16.9

    if hi_teor > hi_prac:
        return 'независимы'
    else:
        return 'зависимы'


def task2(petal_l, petal_w):
    print("\n№3.2.")
    m_len = petal_l.mean()
    m_width = petal_w.mean()

    xy = np.sum(np.multiply(petal_l, petal_w))
    xy_m = xy / 50

    s_len = np.sum(np.power(np.subtract(petal_l, m_len), 2))
    s_len_m = s_len / 50

    s_width = np.sum(np.power(np.subtract(petal_w, m_width), 2))
    s_width_m = s_width / 50

    std_dev_len = np.sqrt(s_len_m)
    std_dev_width = np.sqrt(s_width_m)

    c = xy_m - m_len * m_width
    r = c / (std_dev_len * std_dev_width)

    print("Коэффициент корреляции:", r)
    print("Коэффициент ковариации:", c)

    T = (r / np.sqrt(1 - r ** 2)) * (np.sqrt(50 - 2))
    tss = s_width

    T_table = 1.96

    if T > T_table:
        print("Коэффициент корреляции значим")
    else:
        print("Коэффициент корреляции не значим")
    return xy_m, m_len, m_width, tss, r


def task3(petal_l, petal_w, xy_m, m_len, m_width, tss, r):
    print("\n№3.3.")

    sq_len = np.sum(np.power(petal_l, 2))
    beta1_lid = (xy_m - m_len * m_width) / ((sq_len / 50) - (m_len ** 2))
    beta0_lid = m_width - beta1_lid * m_len

    rss = 0
    ess = 0
    for i in range(50):
        y_lid = beta0_lid + beta1_lid * petal_l.values[i]

        rss += (petal_w.values[i] - y_lid) ** 2
        ess += (y_lid - m_width) ** 2

    print("tss", tss)
    print("rss", rss)
    print("ess", ess)
    determ_ko = 1 - rss / tss
    print("Коэффициент детерминации по формуле 1 - rss/tss", determ_ko)

    determ_ko2 = ess / tss
    print("Коэффициент детерминации по формуле ess/tss", determ_ko2)
    determ_ko3 = r * r
    print("Коэффициент детерминации по формуле r*r", determ_ko3)

    f = (determ_ko3 / (1 - determ_ko3)) * ((50 - 1 - 1) / 1)
    f_table = 4.03
    if f_table > f:
        print("регрессия считается незначимой")
    else:
        print("регрессия считается значимой")


def main():
    df = pd.read_csv("iris.data", delimiter=',')
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = df[df['class'].isin(['Iris-versicolor'])]

    petal_l = df['petal_length']
    petal_w = df['petal_width']

    print("№3.1.\nпеременные", task1(petal_l, petal_w))

    xy_m, m_len, m_width, tss, r = task2(petal_l, petal_w)
    task3(petal_l, petal_w, xy_m, m_len, m_width, tss, r)


if __name__ == '__main__':
    main()
