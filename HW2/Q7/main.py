from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, classes, file_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)


def load_csv(filename):
    dataset = list()
    is_first = True
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if is_first:
                is_first = False
                continue
            dataset.append(row)
    return dataset


def preprocess_data(dataset):
    grouped = dict()
    for row in dataset:
        for col in range(len(row)):
            if col == len(row) - 1:
                row[col] = int(row[col].strip())
            else:
                row[col] = float(row[col].strip())
        class_of_group = row[-1]
        if (class_of_group not in grouped):
            grouped[class_of_group] = list()
        grouped[class_of_group].append(row)
    classes_metadata = dict()
    for class_value, rows in grouped.items():
        class_metadata = [(mean(column), stdev(column), len(column)) for column in zip(*rows)]
        del (class_metadata[-1])
        classes_metadata[class_value] = class_metadata
    return classes_metadata


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_accuracy_metric(real_value, predicted):
    correct = 0
    for i in range(len(real_value)):
        if real_value[i] == predicted[i]:
            correct += 1
    return correct / float(len(real_value)) * 100.0


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def calculate_class_probabilities(classes_metadata, row):
    total_rows = sum([classes_metadata[label][0][2] for label in classes_metadata])
    probabilities = dict()
    for class_value, class_metada in classes_metadata.items():
        probabilities[class_value] = classes_metadata[class_value][0][2] / float(total_rows)
        for i in range(len(class_metada)):
            mean, stdev, _ = class_metada[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def predict(classes_metadata, row):
    probabilities = calculate_class_probabilities(classes_metadata, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def naive_bayse(filename):
    dataset = load_csv(filename)
    classes_metadata = preprocess_data(dataset)

    n_folds = 5
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    recalls = list()
    precisions = list()
    accuracies = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        # Naive Bayes Algorithm
        predicted = list()
        for row in test_set:
            result = predict(classes_metadata, row)
            predicted.append(result)

        actual = [row[-1] for row in fold]
        conf_mtx = confusion_matrix(actual, predicted)
        accuracy = calculate_accuracy_metric(actual, predicted)
        scores.append(accuracy)
        # confusion matrix plot
        plot_confusion_matrix(conf_mtx, ["Actual", "Predicted"], "naive_bayes_confusion_matrix.png")

        FP = conf_mtx.sum(axis=0) - np.diag(conf_mtx)
        FN = conf_mtx.sum(axis=1) - np.diag(conf_mtx)
        TP = np.diag(conf_mtx)
        TN = conf_mtx.sum() - (FP + FN + TP)

        # Recall
        TPR = (TP / (TP + FN))
        recalls.append((TPR[0]) * 100)

        # Precision
        PPV = TP / (TP + FP)
        precisions.append((PPV[0]) * 100)

        # Accuracy
        Accuracy = (TP + TN) / (TP + FN + FP + TN)
        accuracies.append((Accuracy[0]) * 100)

    print('Mean Recalls: %.3f%%' % (sum(recalls) / float(len(recalls))))
    print('Mean Precisions: %.3f%%' % (sum(precisions) / float(len(precisions))))
    print('Mean Accuracy: %.3f%%' % (sum(accuracies) / float(len(accuracies))))



def optimal_bayes(filename):
    data = pd.read_csv(filename)
    X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = data['target']
    

seed(1)
filename = 'Breast_cancer_data.csv'
optimal_bayes(filename)
# naive_bayse()


