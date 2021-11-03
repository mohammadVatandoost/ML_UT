
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics

def pre_process(path):
    data = pd.read_csv(path)
    # columns = data.
    # print(columns)
    # random.shuffle(data.)
    # data.drop(columns=data.columns[0],
    #         axis=1,
    #         inplace=True)
    train_data = data[: int(len(data)*0.7)]
    test_data = data[int(len(data)*0.7): int(len(data)*0.9)]
    evaluation_data = data[int(len(data)*0.9):]
    # print(train_data)
    # print(test_data)
    # print(evaluation_data)
    # print(evaluation_data["TV"])
    # print(evaluation_data["Radio"])
    # print(evaluation_data["Social Media Influencer"])
    # print(evaluation_data[["Sales"]])
    # x = [evaluation_data["TV"], evaluation_data["Radio"], evaluation_data["Social Media Influencer"]]
    return train_data, test_data, evaluation_data


def liner_regression(x, y, degree):
    regressor = PolynomialFeatures(degree=degree)
    X_poly = regressor.fit_transform(x)
    pol_reg = LinearRegression()
    model = pol_reg.fit(X_poly, y)
    return model


def test_model(model, x, y, degree):
    regressor = PolynomialFeatures(degree=degree)
    X_poly = regressor.fit_transform(x)
    pred_y = model.predict(X_poly)
    mse = metrics.mean_squared_error(y, pred_y)
    print("Mean Squared Error {}".format(mse))


if __name__ == '__main__':
    train_data, test_data, evaluation_data = pre_process("./Dummy Data HSS.csv")
    # , "Social Media Influencer"
    model = liner_regression(train_data[["TV", "Radio"]], train_data[["Sales"]], 2)
    print("model")
    print(model.coef_)

    test_model(model, test_data[["TV", "Radio"]], test_data[["Sales"]], 2)


