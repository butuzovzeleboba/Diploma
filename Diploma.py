import math
import numpy as np
import pandas as pd
import statistics
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1)загрузка данных
old_data = pd.read_excel('./old_data_full.xlsx')
old_data.head()
# print(old_data)

new_data = pd.read_excel('./new_data.xlsx')
new_data.head()
# print(new_data)

# 2)стандартизация данных
standart_data_old = StandardScaler(old_data)
print(standart_data_old)
