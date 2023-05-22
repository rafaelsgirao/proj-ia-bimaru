import numpy as np
from sys import exit


def calc_adjacent_sum(m1, m2):
    m_sums = np.empty_like(m1)
    m2_pad = np.pad(m2, pad_width=1, mode="constant", constant_values=0)
    #  print(m2_pad)
    m_filter = np.array(([1, 1, 1], [1, 0, 1], [1, 1, 1]))
    for (i, j), _ in np.ndenumerate(m_sums):
        # m1_around_point = m1[i: i+ 2, j: j+2]
        m2_around_point = m2_pad[i : i + 3, j : j + 3]  # 3x3 matrix around point
        print(f"({i}, {j})")
        print(m2_around_point)
        print("---")
        m_sums[i, j] = np.sum(m2_around_point * m_filter) * m1[i, j] + m1[i, j]

    return np.any(m_sums > 1)  # rg FIXME acabar esta condicao
    return m_sums


# m1 e m2 sem conflito

m1 = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

m2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]])

# m3 e m4 em conflito

m3 = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

m4 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]])
