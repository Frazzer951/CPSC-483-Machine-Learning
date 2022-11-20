from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from utils import load_data, preprocess, progressbar, split_data


def similarity(test_word_set: set, training_word_set: set):
    return len(test_word_set.intersection(training_word_set))


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

    training_sets = []
    for text in training_data:
        training_sets.append(set(text.split()))

    for i in progressbar(range(len(test_data)), "Processing Test Data: "):
        text = test_data[i]
        similarities = []
        test_set = set(text.split())

        for i in range(len(training_data)):
            sim = similarity(test_set, training_sets[i])
            similarities.append([training_labels[i], sim])

        # Sort by max similarity
        similarities = sorted(similarities, key=lambda i: i[1], reverse=True)

        selected_values = []
        for i in range(K):
            selected_values.append(similarities[i])

        result.append(get_class(selected_values))

    return result


def given(word, spam, training_data, training_labels):
    count = 0
    if spam == True:
        for x in range(len(training_labels)):
            if training_labels[x] == True:

                for text in training_data[x]:
                    if word == text:
                        count += 1

    else:
        for x in range(len(training_labels)):
            if training_labels[x] == False:

                for text in training_data[x]:
                    if word == text:
                        count += 1

    return count


def p_w_spam(word, training_data, training_labels, SpamWordCount, Nvoc):
    if word in training_data:

        return (given(word, True, training_data, training_labels) + 1) / (SpamWordCount + alpha * Nvoc)
    else:
        return 1


def p_w_ham(word, training_data, training_labels, NonSpamWordCount, Nvoc):
    if word in training_data:
        return (given(word, False, training_data, training_labels) + 1) / (NonSpamWordCount + 1 * Nvoc)
    else:
        return 1


def classify(message, ProbSpam, ProbNotSpam, training_data, training_labels, SpamWordCount, NonSpamWordCount, Nvoc):
    p_spam_given_message = ProbSpam
    p_ham_given_message = ProbNotSpam
    for word in message:
        p_spam_given_message *= p_w_spam(word, training_data, training_labels, SpamWordCount, Nvoc)
        p_ham_given_message *= p_w_ham(word, training_data, training_labels, NonSpamWordCount, Nvoc)
    if p_ham_given_message > p_spam_given_message:
        return "Not Spam"
    elif p_ham_given_message < p_spam_given_message:
        return "Spam"
    else:
        return "needs human classification"


def main(training_data, test_data, training_labels, test_labels, K):
    # result = knn_classifier(training_data, training_labels, test_data, K)
    # accuracy = accuracy_score(test_labels, result)

    # print(f"training data size\t: {len(training_data)}")
    # print(f"test data size\t\t: {len(test_data)}")
    # print(f"K value\t\t\t: {K}")
    # print(f"% accuracy\t\t: {accuracy * 100}")
    # print(f"Number correct\t\t: {int(accuracy * len(test_data))}")
    # print(f"Number wrong\t\t: {int((1 - accuracy) * len(test_data))}")

    # Naïve Bayes Starts Here
    # Define numbers needed later
    ProbSpam = training_labels.sum() / len(training_labels)
    ProbNotSpam = 1 - ProbSpam
    alpha = 1

    SpamWordCount = 0
    NonSpamWordCount = 0

    for i in range(len(training_data)):
        if training_labels[i] == True:
            text = training_data[i]
            spam_set = set(text.split())
            SpamWordCount += len(spam_set)

        else:
            text = training_data[i]
            notspam_set = set(text.split())
            NonSpamWordCount += len(notspam_set)

    Nvoc = SpamWordCount + NonSpamWordCount

    result = []
    x = 0
    for message in training_data:

        print(x)
        x += 1
        result.append(
            classify(message, ProbSpam, ProbNotSpam, training_data, training_labels, SpamWordCount, NonSpamWordCount, Nvoc)
        )

    print(result)


def find_k(training_data, test_data, training_labels, test_labels):
    accuracies = []
    for K in range(1, 50, 2):
        print(f"\n\nTesting K = {K}")
        result = knn_classifier(training_data, training_labels, test_data, K)
        accuracy = accuracy_score(test_labels, result)
        accuracies.append([K, accuracy * 100])
        print(f"K value\t\t: {K}")
        print(f"% accuracy\t: {accuracy * 100}")
        print(f"Number correct\t: {int(accuracy * len(test_data))}")
        print(f"Number wrong\t: {int((1 - accuracy) * len(test_data))}")

    accuracies_sorted = sorted(accuracies, key=lambda i: i[1], reverse=True)
    pprint(accuracies_sorted)
    print("MAX: " + str(max(accuracies_sorted, key=lambda i: i[1])))

    # plot
    K_accuracy = np.array(accuracies)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]

    plt.figure()
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("% Accuracy")
    plt.title("KNN Algorithm Accuracy With Different K")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    print("Loading Data")
    data = load_data()
    # print(data)

    print("Preprocessing Data")
    data = preprocess(data)
    # print(data)

    print("Spliting Data")
    training_data, test_data, training_labels, test_labels = split_data(data, 0.50)

    # More accurate with a training/test split of 75/25 and a K of 13

    main(training_data, test_data, training_labels, test_labels, 9)
    # find_k(training_data, test_data, training_labels, test_labels)
