import cv2
import glob
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from math import log
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic


def calculate_bic(n, mse, num_params):
   bic = n * log(mse) + num_params * log(n)
   return bic


def em(k):
    file_path = 'Images/*.jp*g'
    files = glob.glob(file_path)
    num_img = 122
    cnt = 0
    arr_blue = []
    arr_green = []
    arr_red = []
    arr_target = []

    for img, file in enumerate(files):
        img = cv2.imread(file)
        #       print(file)
        cnt = cnt + 1
        if (cnt > num_img):
            break

        # average of Blue,Green and red colors
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        # push data of images in array
        arr_blue.append(avg_color[0])
        arr_green.append(avg_color[1])
        arr_red.append(avg_color[2])

        # if avg_color[2]>avg_color[0] probebly it is manchester team
        if (avg_color[2] > avg_color[0]):
            arr_target.append("Manchester")
        else:
            arr_target.append("Chelsea")

    # creat dataframe
    team_dict = {
        'blue': arr_blue,
        'green': arr_green,
        'red': arr_red,
        'target': arr_target}
    data_framwork = pd.DataFrame(team_dict)

    X_features = data_framwork[['blue', 'red']]
    X_features = X_features.to_numpy()
    y_real = data_framwork['target']
    label_encoder = LabelEncoder()
    y_real = label_encoder.fit_transform(y_real)

    GMM = GaussianMixture(n_components=k, covariance_type='full').fit(X_features)  # Instantiate and fit the model
    print('Converged:', GMM.converged_)  # Check if the model has converged
    means = GMM.means_
    covariances = GMM.covariances_
    y_hat = GMM.predict(X_features)
    print('\u03BC = ', means, sep="\n")
    print('\u03A3 = ', covariances, sep="\n")

    x, y = np.meshgrid(np.sort(X_features[:, 0]), np.sort(X_features[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax0 = fig.add_subplot(111)

    ax0.scatter(X_features[:, 0], X_features[:, 1])
    for m, c in zip(means, covariances):
        multi_normal = multivariate_normal(mean=m, cov=c)
        ax0.contour(np.sort(X_features[:, 0]), np.sort(X_features[:, 1]),
                    multi_normal.pdf(XY).reshape(len(X_features), len(X_features)), colors='green', alpha=0.3)
        ax0.scatter(m[0], m[1], c='red', zorder=10, s=100)

    mse = mean_squared_error(y_real, y_hat)
    print('MSE: %.3f' % mse)
    bic = calculate_bic(len(y), mse, k)
    aic = calculate_aic(len(y), mse, k)
    plt.savefig("component_" + str(k) + ".jpeg")
    # plt.show()

    return aic, bic


if __name__ == '__main__':
    k_component = [1, 2, 3, 4, 5]
    all_aic = []
    all_bic = []
    for k in k_component:
        print("======= "+str(k)+" =======")
        aic, bic = em(k)
        all_aic.append(abs(aic))
        all_bic.append(abs(bic))

    barWidth = 0.3
    fig, ax = plt.subplots()
    rects1 = ax.bar(np.arange(len(k_component)), all_aic, color='white', width=barWidth, edgecolor='black', capsize=7,
                    label='AIC')
    rects2 = ax.bar(np.arange(len(k_component))+barWidth, all_bic, color='white', width=barWidth, edgecolor='black', capsize=7,
                    label='BIC', hatch="||")
    ax.set_xlabel('K')
    ax.set_xticks(np.arange(len(k_component)))
    ax.set_xticklabels(k_component)
    ax.legend()
    fig.savefig("AIC_BIC_Comparision.jpeg")
    plt.show()





