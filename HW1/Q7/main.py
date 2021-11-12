
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt



def pre_process(path):
    data = pd.read_csv(path)
    # columns = data.
    # print(columns)
    # random.shuffle(data.)
    # data.drop(columns=data.columns[0],
    #         axis=1,
    #         inplace=True)
    label_encoder = LabelEncoder()
    data["Influencer"] = label_encoder.fit_transform(data["Influencer"])
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
    y_var = np.var(np.abs(y - pred_y))
    mse = metrics.mean_squared_error(y, pred_y)
    # print("Mean Squared Error {}".format(mse))
    return mse, y_var

def show_MSE(polynomial_degree, mse_tests, mse_evaluations):
    fig, ax = plt.subplots()
    plt.plot(polynomial_degree, mse_tests, label="test data")
    plt.plot(polynomial_degree, mse_evaluations, label="evaluation data")
    ax.set_ylabel('MSE', fontsize=32)
    ax.set_xlabel('Model Degree', fontsize=32)
    fig.set_size_inches(18, 9)
    fig.tight_layout()
    fig.savefig("MSE.png", dpi=100, pad_inches=0)
    plt.legend()
    plt.show()


def show_test_parameters(polynomial_degree, mse_tests, bias_tests, variance_tests):
    fig, ax = plt.subplots()
    plt.plot(polynomial_degree, mse_tests, label="MSE")
    plt.plot(polynomial_degree, bias_tests, label="Bias")
    plt.plot(polynomial_degree, variance_tests, label="Variance")
    ax.set_ylabel('MSE', fontsize=32)
    ax.set_xlabel('Model Degree', fontsize=32)
    fig.set_size_inches(18, 9)
    fig.tight_layout()
    fig.savefig("test_parameters.png", dpi=100, pad_inches=0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_data, test_data, evaluation_data = pre_process("./Dummy Data HSS.csv")
    polynomial_degree = [1, 2, 3, 4, 5, 6, 7]
    mse_tests = []
    bias_tests = []
    variance_tests = []
    mse_evaluations = []
    for degree in polynomial_degree:
        model = liner_regression(train_data[["TV", "Radio", "Social Media", "Influencer"]], train_data[["Sales"]], degree)
        mse_test, variance = test_model(model, test_data[["TV", "Radio", "Social Media", "Influencer"]], test_data[["Sales"]], degree)
        mse_evaluate, _ = test_model(model, evaluation_data[["TV", "Radio", "Social Media", "Influencer"]], evaluation_data[["Sales"]], degree)
        print("Degree: {}, mse_test: {}, mse_evaluation: {}, bias: {}, variance: {}".format(degree, mse_test,mse_evaluate, model.intercept_[0], variance))
        mse_tests.append(mse_test)
        mse_evaluations.append(mse_evaluate)
        bias_tests.append(model.intercept_[0])
        variance_tests.append(variance)
        # print("model")
        # print(model.coef_)
        print("=================")

    show_MSE(polynomial_degree, mse_tests, mse_evaluations)
    show_test_parameters(polynomial_degree, mse_tests, bias_tests, variance_tests)









