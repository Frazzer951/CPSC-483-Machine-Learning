import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import assessment_scores, load_data, preprocess, split_data


def score(text, word_occurrence):
    # For each word in text
    words = text.split()
    positive_sum = negative_sum = 0
    for word in words:

        # Sum the number times every word occurs in positive and negative sentiments
        if word in word_occurrence.keys():
            positive_sum += word_occurrence[word][0]
            negative_sum += word_occurrence[word][1]

    # Return the sums and the number of words in the text
    return [positive_sum, negative_sum, len(words)]


def svm_classifier(training_data, training_labels, test_data):
    # Create a dictionary key for every word in the training data
    # and store as a pair the number of times that word occurs in the training data
    word_occurrence = {}
    for i, text in enumerate(training_data):
        words = text.split()
        for word in words:
            if word not in word_occurrence.keys():
                word_occurrence[word] = [0, 0]
            if training_labels[i] == 1:
                word_occurrence[word][0] += 1
            else:
                word_occurrence[word][1] += 1
    # print("Made dictionary")

    # Transform the training data into vectors
    transformed_data = []
    for text in training_data:
        transformed_data.append(score(text, word_occurrence))
    # print("Transformed data")

    # Run SVM on the transformed data
    y = np.array(training_labels)
    X = np.array(transformed_data)
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", max_iter=1000))
    model.fit(X, y)
    # print("Created Model")

    # Transform the test data and classify each one with model
    result = []
    for text in test_data:
        test_vector = np.array([score(text, word_occurrence)])
        result.append(model.predict(test_vector))
    # print("Made predictions")

    return result


def main(training_data, test_data, training_labels, test_labels):
    result = svm_classifier(training_data, training_labels, test_data)

    assessment_scores(test_labels, result)


if __name__ == "__main__":
    print("Loading Data")
    data = load_data()
    # print(data)

    print("Preprocessing Data")
    data = preprocess(data)
    # print(data)

    print("Spliting Data")
    training_data, test_data, training_labels, test_labels = split_data(data, 0.50)

    main(training_data, test_data, training_labels, test_labels)
