import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('canada_per_capita_income.csv')
print(df.tail())

model = linear_model.LinearRegression()
model.fit(df[['year']], df.income)

plt.scatter(df.year, df.income, color='black')
plt.xlabel('Year')
plt.ylabel('Income, CAD')
plt.title('Income per Capita for Canada 1970 - 2016')

plt.plot(df.year, model.predict(df[['year']]), color='black')
plt.show()

print(f"Canadia per capita predicted Income in 2020, CAD: {model.predict([[2020]])}")