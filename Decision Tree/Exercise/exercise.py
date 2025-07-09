import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv('titanic.csv')

target = df['Survived']
print(target.head())

df_real = df.drop(['Survived', 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_real.Sex = df_real.Sex.map({'male': 0, 'female': 1})

print(df_real.head())

X_train, X_test, y_train, y_test = train_test_split(df_real, target, test_size=0.5)

model = DecisionTreeClassifier(criterion='entropy')

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='rocket_r')
plt.show()