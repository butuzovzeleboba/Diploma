import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pingouin as pg
import scipy.stats as stats
import scipy
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import linkage, dendrogram

# 1)загрузка данных и оисательная статистика
LSTU_old = pd.read_excel('./LSTU_old.xlsx')
describe_LSTU_old = LSTU_old.describe()
print(describe_LSTU_old.to_string())
# print(LSTU_old)

LSTU_new = pd.read_excel('./LSTU_new.xlsx')
describe_LSTU_new = LSTU_new.describe()
print(describe_LSTU_new.to_string())
# print(LSTU_new)

VSU_data = pd.read_excel('./VSU_data.xlsx')
describe_VSU_data = VSU_data.describe()
print(describe_VSU_data.to_string())
# print(VSU_data)

# \\\\\\\\\\\\\\\\\\\\\\\
# 2)стандартизация данных
object1 = StandardScaler()
standart_LSTU_old = pd.DataFrame(object1.fit_transform(LSTU_old))
# print(standart_LSTU_old)

object2 = StandardScaler()
standart_LSTU_new = pd.DataFrame(object2.fit_transform(LSTU_new))
# print(standart_LSTU_new)

object3 =StandardScaler()
standart_VSU_data = pd.DataFrame(object3.fit_transform(VSU_data))
# print(standart_VSU_data)
# \\\\\\\\\\\\\\\\\\\\
# 3)Матрица корреляции
matrix_LSTU_old = LSTU_old.corr()
sns.heatmap(matrix_LSTU_old, vmin=-0.5, vmax=1, annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_LSTU_old.shape[1], k=-1, dtype=bool),
linewidth =2, cbar=False)
plt.show()

matrix_LSTU_new = LSTU_new.corr()
sns.heatmap(matrix_LSTU_new, vmin=-0.5, vmax=1, annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_LSTU_new.shape[1], k=-1, dtype=bool),
linewidth =2, cbar=False)
plt.show()

matrix_VSU_data = VSU_data.corr()
sns.heatmap(matrix_VSU_data, vmin=-0.5, vmax=1, annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_VSU_data.shape[1], k=-1, dtype=bool),
linewidth =2, cbar=False)
plt.show()

# 5)Метод локтя
X1 = standart_LSTU_new
wcss1=[]
for i in range(1, 12):
    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=8, random_state=12)
    k_means.fit(X1)
    wcss1.append(k_means.inertia_)
plt.plot(range(1, 12), wcss1)
plt.title('Метод локтя ЛГТУ новые данные')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

X2 = standart_LSTU_old
wcss2=[]
for i in range(1, 12):
    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=8, random_state=12)
    k_means.fit(X2)
    wcss2.append(k_means.inertia_)
plt.plot(range(1, 12), wcss2)
plt.title('Метод локтя ЛГТУ старые данные')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

X3 = standart_VSU_data
wcss3=[]
for i in range(1, 12):
    k_means = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=8, random_state=12)
    k_means.fit(X3)
    wcss3.append(k_means.inertia_)
plt.plot(range(1, 12), wcss3)
plt.title('Метод локтя ВГУ')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

# \\\\\\\\\\\\\\\
# 6)Альфа кронбаха
alpha_LSTU_new = pg.cronbach_alpha(data=standart_LSTU_new, ci=.95)
print("Альфа Кронбаха ЛГТУ новые", alpha_LSTU_new)

alpha_LSTU_old = pg.cronbach_alpha(data=standart_LSTU_old, ci=.95)
print("Альфа Кронбаха ЛГТУ старые", alpha_LSTU_old)# Посмотреть точнее

alpha_VSU = pg.cronbach_alpha(data=standart_VSU_data, ci=.95)
print("Альфа Кронбаха ВГУ", alpha_VSU)

#7) Критерий согласия Пирсона
stat, p = scipy.stats.normaltest(LSTU_old['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности по старым данным ЛГТУ')
else:
    print('Отклонить гипотезу о нормальности по старым данным ЛГТУ')

tat, p = scipy.stats.normaltest(LSTU_new['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.044
if p > alpha:
    print('Принять гипотезу о нормальности по новым данным ЛГТУ')
else:
    print('Отклонить гипотезу о нормальности по новым данным ЛГТУ')# Посмотреть точнее

tat, p = scipy.stats.normaltest(VSU_data['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности по данным ВГУ')
else:
    print('Отклонить гипотезу о нормальности по данным ВГУ')

# 8)Линейная регрессия
model = smf.ols('Престиж ~ Востребованность', data=LSTU_old)
res = model.fit()
print(res.summary())

model = smf.ols('Престиж ~ Востребованность', data=LSTU_new)
res = model.fit()
print(res.summary())

model = smf.ols('Престиж ~ Востребованность', data=VSU_data)
res = model.fit()
print(res.summary())

# \\\\\\\\\\\\\\\\\
# 9)Кластеное дерево
samples = LSTU_new.values
varieties1 = list(LSTU_new.pop('Престиж'))
mergings = linkage(samples, method='complete')
dendrogram(mergings,
           labels=varieties1,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

samples = LSTU_old.values
varieties2 = list(LSTU_old.pop('Престиж'))
mergings = linkage(samples, method='complete')
dendrogram(mergings,
           labels=varieties2,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

samples = VSU_data.values
varieties3 = list(VSU_data.pop('Престиж'))
mergings = linkage(samples, method='complete')
dendrogram(mergings,
           labels=varieties3,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

# \\\\\\\\\\\\\\\\\
# # 6)Тест Хи квадрат
# standart_LSTU_new = standart_LSTU_new.values
# hi_LSTU_new, p_value = stats.chisquare(f_obs=standart_LSTU_new)
# print('hi_LSTU_new : ' +
#       str(hi_LSTU_new))
# print('p_value : ' + str(p_value))
# print(hi_LSTU_new)

# # \\\\\\\\\\\
# # 4)К-средних
# X = LSTU_new.values
# Clus_dataSet = StandardScaler().fit_transform(X)
# # print(Clus_dataSet)
# k_means = KMeans(init="k-means++", n_clusters=2, n_init=25)
# k_means.fit(X)
# labels = k_means.labels_
# # print(labels)
# LSTU_new["Clus_km"] = labels
# # print(LSTU_new.head(n=7))
# LSTU_new.groupby('Clus_km').mean()
# area = np.pi * (X[:, 1])**2
# plt.scatter(X[:, 7], X[:, 8], s=area, c=labels.astype(np.float64), alpha=0.5)
# plt.xlabel('Престиж', fontsize=10)
# plt.ylabel('Мнение родственников', fontsize=10)
# plt.show()
# # fig = plt.figure(1, figsize=(8, 6))
# # plt.clf()
# # ax = Axes3D(fig, rect=[0, 0, 95, 1], elev=48, azim=134)
#
# # ax.set_xlabel()
# # ax.set_ylabel()
# # ax.set_zlabel()
#
# # ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float64))


