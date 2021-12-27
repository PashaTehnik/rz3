from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd


def task1(df):
    print("№4.1")
    df = df[(df['class'] == "Iris-virginica") | (df['class'] == "Iris-versicolor")]
    df = df.drop(["sepal_length", "petal_length", "sepal_width"], axis=1)
    x = df.drop(['class'], axis=1)
    y = df['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(x_train, y_train)
    print("модель обучена")
    return model, x_train, x_test, y_train, y_test


def task2(model, x_test, y_test):
    print("\n\n№4.2")
    predicted = model.predict(x_test)
    print("точность = ", accuracy_score(predicted, y_test), "\n")
    for i in range(len(x_test)):
        print(predicted[i] == y_test.values[i], predicted[i], y_test.values[i])


def main():
    df = pd.read_csv("iris.data", delimiter=',')
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    model, x_train, x_test, y_train, y_test = task1(df)

    task2(model, x_test, y_test)


if __name__ == '__main__':
    main()
