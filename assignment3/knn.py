from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from utils import load_data, preprocess, progressbar, split_data


def get_count(text):
    word_counts = defaultdict(int)
    for word in text.split():
        word_counts[word] += 1
    return word_counts


def difference(test_counts, training_counts):
    """Calculate the Euclidian difference between two word counts"""
    total = 0
    for word in test_counts:
        if word in training_counts:  # If word is in both counts add the square of the difference in frequency
            total += (test_counts[word] - training_counts[word]) ** 2
            del training_counts[word]
        else:  # If word is only in test_counts, add the square of its frequency
            total += test_counts[word] ** 2
    for word in training_counts:  # If word is only in training_counts, add the square of its frequency
        total += training_counts[word] ** 2
    return total**0.5  # return the sqrt of the total


def get_class(selected_values):
    spam_count = 0
    real_count = 1

    for value in selected_values:
        if value[0] == 1:
            spam_count += 1
        else:
            real_count += 1
    return spam_count > real_count


def knn_classifier(training_data, training_labels, test_data, K):
    result = []

    training_counts = []
    for text in training_data:
        training_counts.append(get_count(text))

    for i in progressbar(range(len(test_data)), "Processing Test Data: "):
        text = test_data[i]
        similarity = []
        test_counts = get_count(text)

        for i in range(len(training_data)):
            diff = difference(test_counts, training_counts[i])
            similarity.append([training_labels[i], diff])

        # Sort by difference
        similarity = sorted(similarity, key=lambda i: i[1])

        selected_values = []
        for i in range(K):
            selected_values.append(similarity[i])

        result.append(get_class(selected_values))

    return result


def main(K):
    print("Loading Data")
    data = load_data()
    # print(data)

    print("Preprocessing Data")
    data = preprocess(data)
    # print(data)

    print("Spliting Data")
    training_data, test_data, training_labels, test_labels = split_data(data, 0.25)

    # print("Training Data")
    # print(training_data)
    # print("Training Labels")
    # print(training_labels)

    # print("Test Data")
    # print(test_data)
    # print("Test Labels")
    # print(test_labels)

    result = knn_classifier(training_data, training_labels, test_data, K)
    accuracy = accuracy_score(test_labels, result)

    print(f"training data size\t: {len(training_data)}")
    print(f"test data size\t\t: {len(test_data)}")
    print(f"K value\t\t\t: {K}")
    print(f"% accuracy\t\t: {accuracy * 100}")
    print(f"Number correct\t\t: {int(accuracy * len(test_data))}")
    print(f"Number wrong\t\t: {int((1 - accuracy) * len(test_data))}")


def find_k():
    print("Loading Data")
    data = load_data()
    print("Preprocessing Data")
    data = preprocess(data)
    print("Spliting Data")
    training_data, test_data, training_labels, test_labels = split_data(data, 0.25)

    accuracies = []
    for K in range(1, 25, 2):
        result = knn_classifier(training_data, training_labels, test_data, K)
        accuracy = accuracy_score(test_labels, result)
        accuracies.append([K, accuracy * 100])
        print(f"K value\t\t\t: {K}")
        print(f"% accuracy\t\t: {accuracy * 100}")
        print(f"Number correct\t\t: {int(accuracy * len(test_data))}")
        print(f"Number wrong\t\t: {int((1 - accuracy) * len(test_data))}")

    accuracies_sorted = sorted(accuracies, key=lambda i: i[1])
    print(accuracies_sorted)
    print("MAX: " + str(max(accuracies_sorted, key=lambda i: i[1])))

    # plot
    K_accuracy = np.array(accuracies)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]

    plt.figure()
    plt.ylim(0, 101)
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("% Accuracy")
    plt.title("KNN Algorithm Accuracy With Different K")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # main(5)
    find_k()
