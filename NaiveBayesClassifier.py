# Gagandeep Batra
# Naive Bayes Classifier for Buy (categorical) and Iris (continuous) data sets
# 3/18/2021

import numpy as np
import math

# Buy Files
buyTesting = np.loadtxt("buyTesting.txt")
buyTraining = np.loadtxt("buyTraining.txt")

# Iris Files
irisTesting = np.loadtxt("irisTesting.txt")
irisTraining = np.loadtxt("irisTraining.txt")


class Continuous:
    # Get mean
    def mean(self, data):
        return np.mean(data)

    # Get standard deviation
    def standard_dev(self, data):
        return np.std(data)

    # Get normal distribution
    def pdf(self, x, mean, sd):
        var = float(sd) ** 2
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    # Calculate probability for each label
    def naive_bayes_calc(self, summarized_data, new_instance, size):
        label_probabilties = dict()
        for label, data in summarized_data.items():
            label_count = summarized_data[label][0][2]
            # get P(C+) and P(C-)
            label_probabilties[label] = label_count / size
            for i in range(len(data)):
                mean, sd = data[i][:2]
                # get P(Xi|C+) and P(Xi|C-)
                label_probabilties[label] *= self.pdf(new_instance[i], mean, sd)
        return label_probabilties

    # Separate data by class label
    def separate_class_label(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            instance = dataset[i]
            label = instance[-1]
            if label not in separated:
                separated[label] = list()
            separated[label].append(instance)
        return separated

    # Calculate mean and standard deviation for each attribute
    def summarize_attribute_data(self, dataset):
        return [(self.mean(attribute), self.standard_dev(attribute), len(attribute)) for attribute in zip(*dataset)][
               :len(dataset[0]) - 1]

    # Calculate mean and standard deviation for each class
    def summarize_class_data(self, dataset):
        separated = self.separate_class_label(dataset)
        data = dict()
        for k, v in separated.items():
            data[k] = self.summarize_attribute_data(v)
        return data

    # Use the training data to predict class label
    def predicted_actual_labels(self, summarized_training, testing_data):
        labels = []
        size = float(testing_data.shape[0])
        for i in range(len(testing_data)):
            predicted_label = self.predict_class_label(
                self.naive_bayes_calc(summarized_training, testing_data[i], size))
            labels.append(predicted_label)
            labels.append(testing_data[i][-1])
        return np.reshape(labels, (testing_data.shape[0], 2))

    # Get class label from max probability
    def predict_class_label(self, naive_bayes_prob):
        max_prob = max(naive_bayes_prob.values())
        for k, v in naive_bayes_prob.items():
            if v == max_prob:
                return k


class Categorical:
    def __init__(self, training_data, testing_data):
        training_labels = list(training_data[:, -1])
        self.testing_labels = list(testing_data[:, -1])
        self.label_count = {
            1: training_labels.count(1),
            -1: training_labels.count(-1)
        }

    # Separate data by class label and category
    def categorical_separate_class_training(self, training_data):
        class_data = dict()
        for i in range(len(training_data)):
            for k in range(len(training_data[0]) - 1):
                if k not in class_data:
                    class_data[k] = list()
                class_data[k].append((training_data[i][k], training_data[i][-1]))
        return class_data

    # Separate labels by probability
    def prob_separate_label_training(self, separate_training_data):
        labels_prob = dict()
        for k, v in separate_training_data.items():
            for i in v:
                if k not in labels_prob:
                    labels_prob[k] = list()
                prob = (i, v.count(i) / self.label_count.get(i[1]))
                if prob not in labels_prob[k]:
                    labels_prob[k].append(prob)
        return labels_prob

    # Predict class label from summarized training data
    def prob_label_testing(self, summarized_training, testing):
        pos = []
        neg = []
        for instance in testing:
            pos_prob = 1
            neg_prob = 1
            test_instance = instance[:-1]
            for k, v in summarized_training.items():
                test_pos = (test_instance[k], 1)
                test_neg = (test_instance[k], -1)
                for i in range(len(v)):
                    if v[i][0] == test_pos:
                        pos_prob *= v[i][1]
                    if v[i][0] == test_neg:
                        neg_prob *= v[i][1]
            pos.append(pos_prob)
            neg.append(neg_prob)
        label_prob = np.column_stack((pos, neg))
        predict_labels = []
        for x in label_prob:
            if x[0] > x[1]:
                predict_labels.append(1)
            else:
                predict_labels.append(-1)
        return np.column_stack((predict_labels, self.testing_labels))


# Get the confusion matrix for model
def confusion_matrix(labels):
    matrix = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }
    for label in labels:
        if label[0] == 1 and label[1] == 1:
            matrix['TP'] += 1
        elif label[0] == 1 and label[1] == -1:
            matrix['FP'] += 1
        elif label[0] == -1 and label[1] == 1:
            matrix['FN'] += 1
        else:
            matrix['TN'] += 1
    return matrix


# calculate classification accuracy for model
def calculate_class_accuracy(labels):
    correct = 0
    for label in labels:
        if label[0] == label[1]:
            correct += 1
    return correct / len(labels) * 100


# Test for Iris Dataset
iris = Continuous()
summarized_training = iris.summarize_class_data(irisTraining)
predicted_labels = (iris.predicted_actual_labels(summarized_training, irisTesting))
print("Classification Accuracy for Iris Testing Dataset: " + str(calculate_class_accuracy(predicted_labels)) + "%")
print("Confusion Matrix for Iris Testing Dataset: " + str(confusion_matrix(predicted_labels)))

# Test for Buy Dataset
buy = Categorical(buyTraining, buyTesting)
summarized_training = buy.prob_separate_label_training(buy.categorical_separate_class_training(buyTraining))
predicted_labels = buy.prob_label_testing(summarized_training, buyTesting)
print("\nClassification Accuracy for Buy Testing Dataset: " + str(calculate_class_accuracy(predicted_labels)) + "%")
print("Confustion Matrix for Buy Testing Dataset: " + str(confusion_matrix(predicted_labels)))