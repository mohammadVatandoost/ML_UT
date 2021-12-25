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


def em(k, X, y, title, saving_file):
    X_features = X.to_numpy()
    label_encoder = LabelEncoder()
    y = y.to_numpy()
    y_real = label_encoder.fit_transform(y)

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
    plt.title(title)
    plt.savefig(saving_file)
    plt.show()

    return aic, bic


def scatter_plot(x1, x2, y, title, xLabel, yLabel, labels, saving_file):
    fig, ax = plt.subplots()
    colors = ["red", "green", "yellow"]
    for i in range(3):
        # label_encoder = LabelEncoder()
        # y_encoded = label_encoder.fit_transform(y[i])
        ax.scatter(x1[i], x2[i], c=colors[i], label=labels[i])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.legend()
    plt.title(title)
    fig.savefig(saving_file)
    plt.show()


if __name__ == '__main__':
    k_component = [2, 3, 4, 5]
    dataset = pd.read_csv('penguins.csv')
    adelie = dataset.loc[dataset['species'] == "Adelie"]
    chinstrap = dataset.loc[dataset['species'] == "Chinstrap"]
    gentoo = dataset.loc[dataset['species'] == "Gentoo"]
    labels = ["Adelie", "chinstrap", "gentoo"]
    scatter_plot([adelie[['culmen_length_mm']].to_numpy(), chinstrap[['culmen_length_mm']].to_numpy(),
                  gentoo[['culmen_length_mm']].to_numpy()], [adelie[['culmen_depth_mm']].to_numpy(),
                  chinstrap[['culmen_depth_mm']].to_numpy(), gentoo[['culmen_depth_mm']].to_numpy()],
                 [adelie[['species']].to_numpy(), chinstrap[['species']].to_numpy(), gentoo[['species']].to_numpy()],
                 "culmen_length_mm - culmen_depth_mm", "culmen_length_mm", "culmen_depth_mm", labels,
                 "culmen_length_mm - culmen_depth_mm.jpeg")

    scatter_plot([adelie[['flipper_length_mm']].to_numpy(), chinstrap[['flipper_length_mm']].to_numpy(),
                  gentoo[['flipper_length_mm']].to_numpy()], [adelie[['culmen_length_mm']].to_numpy(),
                                                             chinstrap[['culmen_length_mm']].to_numpy(),
                                                             gentoo[['culmen_length_mm']].to_numpy()],
                 [adelie[['species']].to_numpy(), chinstrap[['species']].to_numpy(), gentoo[['species']].to_numpy()],
                 "flipper_length_mm - culmen_length_mm", "flipper_length_mm", "culmen_length_mm", labels,
                 "flipper_length_mm - culmen_length_mm.jpeg")


    scatter_plot([adelie[['body_mass_g']].to_numpy(), chinstrap[['body_mass_g']].to_numpy(),
                  gentoo[['body_mass_g']].to_numpy()], [adelie[['flipper_length_mm']].to_numpy(),
                  chinstrap[['flipper_length_mm']].to_numpy(), gentoo[['flipper_length_mm']].to_numpy()],
                 [adelie[['species']].to_numpy(), chinstrap[['species']].to_numpy(), gentoo[['species']].to_numpy()],
                 "body_mass_g - flipper_length_mm", "body_mass_g", "flipper_length_mm", labels,
                 "body_mass_g - flipper_length_mm.jpeg")


    scatter_plot([adelie[['flipper_length_mm']].to_numpy(), chinstrap[['flipper_length_mm']].to_numpy(),
                  gentoo[['flipper_length_mm']].to_numpy()], [adelie[['culmen_depth_mm']].to_numpy(),
                  chinstrap[['culmen_depth_mm']].to_numpy(), gentoo[['culmen_depth_mm']].to_numpy()],
                 [adelie[['species']].to_numpy(), chinstrap[['species']].to_numpy(), gentoo[['species']].to_numpy()],
                 "flipper_length_mm - culmen_depth_mm", "flipper_length_mm", "culmen_depth_mm", labels,
                 "flipper_length_mm - culmen_depth_mm.jpeg")

    print("======= flipper_length_mm - culmen_depth_mm =======")
    em(3, dataset[['culmen_length_mm', 'culmen_depth_mm']], dataset[['species']],
       "EM_culmen_length_mm - culmen_depth_mm", "EM_culmen_length_mm - culmen_depth_mm.jpeg")
    print("======= flipper_length_mm', 'culmen_length_mm =======")
    em(3, dataset[['flipper_length_mm', 'culmen_length_mm']], dataset[['species']],
       "EM flipper_length_mm - culmen_length_mm", "EM flipper_length_mm - culmen_length_mm.jpeg")
    print("======= body_mass_g', 'culmen_length_mm =======")
    em(3, dataset[['body_mass_g', 'flipper_length_mm']], dataset[['species']],
       "EM body_mass_g - flipper_length_mm", "EM body_mass_g - flipper_length_mm.jpeg")
    print("======= flipper_length_mm', 'culmen_depth_mm =======")
    em(3, dataset[['flipper_length_mm', 'culmen_depth_mm']], dataset[['species']],
       "EM flipper_length_mm - culmen_depth_mm", "EM flipper_length_mm - culmen_depth_mm.jpeg")

    all_aic = []
    all_bic = []
    for k in k_component:
        print("======= " + str(k) + " =======")
        aic, bic = em(3, dataset[['culmen_length_mm', 'culmen_depth_mm']], dataset[['species']],
       "EM_culmen_length_mm - culmen_depth_mm", "EM_culmen_length_mm - culmen_depth_mm_" + str(k) + "jpeg")
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
    plt.title("AIC-BIC EM_culmen_length_mm - culmen_depth_mm")
    fig.savefig("AIC_BIC_Comparision.jpeg")
    plt.show()


