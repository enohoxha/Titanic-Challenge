import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('/home/manoolia/code/python/kaggle/titanic-challange/input/train.csv')
test_data = pd.read_csv('/home/manoolia/code/python/kaggle/titanic-challange/input/test.csv')

survived = train_data['Survived']
train_data.drop('Survived', axis=1, inplace=True)

model = DecisionTreeClassifier()
model.fit(train_data, survived)
predictions = model.predict(test_data)

print(predictions)

