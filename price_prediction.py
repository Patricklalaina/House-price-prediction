# Enter your code here. Read input from STDIN. Print output to STDOUT
from sklearn.linear_model import LinearRegression
import numpy as np
import sys


def _process(x_train, y_train, x_test):
    model = LinearRegression()
    reg = model.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    return y_predict


def _get_data():
    return sys.stdin.read().strip().splitlines()


def get_idx_begin(data, stop):
    for i in range(len(data)):
        if len(data[i]) == stop and i != 0:
            return i
    return 0


def main():
    data = [lst.split() for lst in _get_data()]
    F, N = int(data[0][0]), int(data[0][1])
    dataset = [[float(i) for i in lst] for lst in data[1:] if len(lst) == F + 1]
    if len(dataset) != N:
        return
    x_test = np.array([[float(i) for i in lst] for lst in data[get_idx_begin(data, F):]]).reshape(-1, F)
    x_train = np.array([i[0:F] for i in dataset])
    y_train = np.array([i[-1] for i in dataset]).reshape(1, -1)[0]
    y_predict = [round(float(i), 2) for i in _process(x_train, y_train, x_test)]
    for elem in y_predict:
        print(elem)


if __name__=='__main__':
    main()
