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
sns.heatmap(matrix_LSTU_old, annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_LSTU_old.shape[1], k=-1, dtype=bool),
linewidth =2)
plt.show()

matrix_LSTU_new = LSTU_new.corr()
sns.heatmap(matrix_LSTU_new,  annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_LSTU_new.shape[1], k=-1, dtype=bool),
linewidth =2)
plt.show()

matrix_VSU_data = VSU_data.corr()
sns.heatmap(matrix_VSU_data, annot=True, fmt='.2f', linewidths=2,
mask=~np.tri(matrix_VSU_data.shape[1], k=-1, dtype=bool),
linewidth =2)
plt.show()

# 5)Метод локтя
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

#7) Тест Шапиро-Вилка
stat, p = scipy.stats.normaltest(LSTU_new['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.044
if p > alpha:
    print('Принять гипотезу о нормальности по новым данным ЛГТУ')
else:
    print('Отклонить гипотезу о нормальности по новым данным ЛГТУ')# Посмотреть точнее

stat, p = scipy.stats.normaltest(LSTU_old['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности по старым данным ЛГТУ')
else:
    print('Отклонить гипотезу о нормальности по старым данным ЛГТУ')

stat, p = scipy.stats.normaltest(VSU_data['Престиж'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности по данным ВГУ')
else:
    print('Отклонить гипотезу о нормальности по данным ВГУ')

# 8)Линейная регрессия
model = smf.ols('Престиж ~ Востребованность', data=LSTU_new)
res = model.fit()
print(res.summary())

model = smf.ols('Престиж ~ Востребованность', data=LSTU_old)
res = model.fit()
print(res.summary())

model = smf.ols('Престиж ~ Востребованность', data=VSU_data)
res = model.fit()
print(res.summary())

# \\\\\\\\\\\\\\\\\
# 9)Кластеное дерево
Clusters_tree_new=LSTU_new.T
mergings = linkage(Clusters_tree_new, method='complete')
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

LSTU_old_copy=LSTU_old.copy()
LSTU_old_copy=LSTU_old_copy.drop('Посещаемость', axis= 1 , inplace= False )
Clusters_tree_old =LSTU_old_copy.T
mergings = linkage(Clusters_tree_old, method='complete')
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

Clusters_tree_VSU=VSU_data.T
mergings = linkage(Clusters_tree_VSU, method='complete')
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=6,
           )
plt.show()

# \\\\\\\\\\\\\\\\\
# 10)Свечи
sns.catplot(x='Престиж',
            y='Востребованность',
            kind='box',
            data=LSTU_new)
plt.show()

sns.catplot(x='Престиж',
            y='Востребованность',
            kind='box',
            data=LSTU_old)
plt.show()

sns.catplot(x='Престиж',
            y='Востребованность',
            kind='box',
            data=VSU_data)
plt.show()
