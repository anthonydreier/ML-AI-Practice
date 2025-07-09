import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

print(iris.feature_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
print(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5)

model = RandomForestClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_pred = model.predict(X_test)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='rocket_r')
plt.show()