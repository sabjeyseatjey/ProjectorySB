import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# 1.  (MEDV)
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['MEDV'])
plt.title('Боксплот медианной стоимости домов')
plt.ylabel('Стоимость (тыс. $)')
plt.show()

# 2.(CHAS)
plt.figure(figsize=(6, 4))
df['CHAS'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Гистограмма реки Чарльз')
plt.xlabel('Наличие реки (0 - нет, 1 - да)')
plt.ylabel('Количество районов')
plt.show()

# 3. Боксплот MEDV vs AGE
plt.figure(figsize=(6, 4))
sns.boxplot(x=pd.cut(df['AGE'], bins=3), y=df['MEDV'])
plt.title('Боксплот стоимости домов по возрасту')
plt.xlabel('Возраст зданий (группы)')
plt.ylabel('Стоимость (тыс. $)')
plt.show()

# 4. Диаграмма рассеяния NOX vs INDUS
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['INDUS'], y=df['NOX'])
plt.title('Концентрация NOX vs Доля неторговых акров')
plt.xlabel('Доля неторговых земель (%)')
plt.ylabel('Концентрация оксида азота')
plt.show()

# 5. Гистограмма для PTRATIO
plt.figure(figsize=(6, 4))
sns.histplot(df['PTRATIO'], bins=10, kde=True)
plt.title('Гистограмма соотношения учеников и учителей')
plt.xlabel('Соотношение учеников и учителей')
plt.ylabel('Частота')
plt.show()

# Нулевая и альтернативная гипотезы
hypotheses = [
    ('H0: Никакой связи между уровнем NOX и долей неторговых земель нет',
     'H1: Есть связь между уровнем NOX и долей неторговых земель'),
    ('H0: Возраст зданий не влияет на стоимость домов',
     'H1: Возраст зданий влияет на стоимость домов'),
    ('H0: Соотношение учеников и учителей не влияет на стоимость жилья',
     'H1: Соотношение учеников и учителей влияет на стоимость жилья'),
    ('H0: Расстояние до центров занятости не влияет на стоимость жилья',
     'H1: Расстояние до центров занятости влияет на стоимость жилья')
]

conclusions = [
    'Корреляция между уровнем NOX и долей неторговых земель положительная, что указывает на связь.',
    'Возраст зданий имеет значительное влияние на медианную стоимость домов, подтверждая альтернативную гипотезу.',
    'Соотношение учеников и учителей также показывает некоторую зависимость от стоимости жилья.'
]
