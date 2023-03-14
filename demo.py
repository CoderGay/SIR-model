# Example4 -- Kendall correlation coefficient
from scipy.stats.stats import kendalltau
import numpy as np
import matplotlib.pyplot as plt

#
# dat1 = np.array([3, 5, 1, 9, 7, 2, 8, 4, 6])
# dat2 = np.array([5, 3, 2, 6, 8, 1, 7, 9, 4])
# fig, ax = plt.subplots()
# ax.scatter(dat1, dat2)
# kendalltau(dat1, dat2)
# # plt.show()
# c = 0
# d = 0
# for i in range(len(dat1)):
#     for j in range(i + 1, len(dat1)):
#         if (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) > 0:
#             c = c + 1
#         else:
#             d = d + 1
# k_tau = (c - d) * 2 / len(dat1) / (len(dat1) - 1)
#
# print('k_tau = {0}'.format(k_tau))
#


dat1 = np.array([3, 5, 1, 6, 7, 2, 8, 8, 4])
dat2 = np.array([5, 3, 2, 6, 8, 1, 7, 8, 4])
# dat1 = np.array([3,5,1,9,7,2,8,4,6])
# dat2 = np.array([5,3,2,6,8,1,7,9,4])
c = 0
d = 0
t_x = 0
t_y = 0
for i in range(len(dat1)):
    for j in range(i + 1, len(dat1)):
        # 统计一致对个数
        if (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) > 0:
            c = c + 1

        # 统计分歧对个数
        elif (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) < 0:
            d = d + 1

        else:
            # 统计数据X中的并列排位个数, 同时发生在X和Y中并列排位则不计入
            if (dat1[i] - dat1[j]) == 0 and (dat2[i] - dat2[j]) != 0:
                t_x = t_x + 1

            # 统计数据Y中的并列排位个数, 同时发生在X和Y中并列排位则不计入
            elif (dat1[i] - dat1[j]) != 0 and (dat2[i] - dat2[j]) == 0:
                t_y = t_y + 1

tau_b = (c - d) / np.sqrt((c + d + t_x) * (c + d + t_y))

print('tau_b = {0}'.format(tau_b))
print('kendalltau(dat1,dat2) =  {0}'.format(kendalltau(dat1, dat2)))
