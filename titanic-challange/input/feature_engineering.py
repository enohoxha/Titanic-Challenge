import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', 500)


def clean_features(data, type):

    df = pd.DataFrame(data)
    # df = df.drop("PassengerId", axis=1)
    df.set_index("PassengerId")
    df = df.drop(columns=['Cabin', 'Name', 'Ticket'])
    print((df.Fare == 0).sum())
    if type == True:
        analyse_data(df)


    return df


def analyse_data(df):
    print(df.Survived.value_counts(normalize=True))
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))

    df["Faily_count"] = df.SibSp + df.Parch
    df["Faily_count"] = pd.cut(df["Faily_count"], bins=[-1, 0, 3, 7, 16], labels=["Alone", "Small Family", "Medium Family", "Big Family"])
    alive_family = df[df["Survived"] == 1]["Faily_count"].value_counts()
    dead_family = df[df["Survived"] == 0]["Faily_count"].value_counts()
    family_df = pd.DataFrame([alive_family, dead_family])
    family_df.index = ['Alive', 'Dead']
    family_df.plot(kind="bar", stacked=True, ax=axes[0][1])




    df["Age_Range"] = pd.cut(df["Age"], bins=[1, 14, 30, 50, 80], labels=["Child", "Adult", "MidAge", "Old"])
    alive_age = df[df["Survived"] == 1]["Age_Range"].value_counts()
    dead_age = df[df["Survived"] == 0]["Age_Range"].value_counts()
    age_df = pd.DataFrame([alive_age, dead_age])
    age_df.index = ['Alive', 'Dead']
    age_df.plot(kind="bar", stacked=True, ax=axes[0][0])
    plt.show()


# #  ['PassengerId', 'Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare',  'Embarked']

def feature_extraction(data, train=True):
    df = pd.DataFrame(data)
    df.set_index("PassengerId")

    df.Fare = df.Fare.replace(0, np.NaN)
    df.Fare.fillna(df.Fare.mean(), inplace=True)
    df.Embarked.fillna("S", inplace=True)
    # todo improve age (do not find it by mean)
    df.Age.fillna(df.Age.mean(), inplace=True)

    df["Age_Range"] = pd.cut(df["Age"], bins=[0, 14, 30, 50, 80], labels=["Child", "Adult", "MidAge", "Old"])
    df["Family_Count"] = df.SibSp + df.Parch
    df["Family_Count"] = pd.cut(df["Family_Count"], bins=[-1, 0, 3, 9, 16],
                               labels=["Alone", "Small Family", "Medium Family", "Big Family"])

    if train:
        df = df[['Survived', 'Pclass', 'Sex', 'Embarked', 'Age_Range', 'Family_Count']]
    else:
        df = df[['PassengerId', 'Pclass', 'Sex',  'Embarked', 'Age_Range', 'Family_Count']]

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "Q": 1, "C": 2})
    df["Age_Range"] = df["Age_Range"].map({"Child": 0, "Adult": 1, "MidAge": 2, "Old": 3})
    df["Family_Count"] = df["Family_Count"].map({"Alone": 0, "Small Family": 1, "Medium Family": 2, "Big Family": 3})

    return df