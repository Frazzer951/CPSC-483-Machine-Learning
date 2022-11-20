from collections import defaultdict


from utils import assessment_scores, load_data, preprocess, progressbar, split_data


def classify(
    message,
    ProbSpam,
    ProbNotSpam,
    spamWordCounts,
    spamWordCount,
    realWordCounts,
    realWordCount,
    allWords,
    allWordsCount,
):
    words = message.split()
    p_spam = ProbSpam
    p_real = ProbNotSpam

    for word in words:
        if word in allWords:
            # Spam
            p_spam *= (spamWordCounts[word] + 1) / (spamWordCount + allWordsCount)
            # Real
            p_real *= (realWordCounts[word] + 1) / (realWordCount + allWordsCount)

    if p_spam > p_real:
        return True
    else:
        return False


def naiveBayes(training_data, test_data, training_labels, test_labels):
    # Naïve Bayes Starts Here
    # Define numbers needed later
    ProbSpam = training_labels.sum() / len(training_labels)
    ProbNotSpam = 1 - ProbSpam

    spamWordCounts = defaultdict(int)
    spamWordCount = 0

    realWordCounts = defaultdict(int)
    realWordCount = 0

    allWords = set()

    for i in range(len(training_data)):
        if training_labels[i]:
            words = training_data[i].split()
            spamWordCount += len(words)
            for word in words:
                spamWordCounts[word] += 1
                allWords.add(word)
        else:
            words = training_data[i].split()
            realWordCount += len(words)
            for word in words:
                realWordCounts[word] += 1
                allWords.add(word)

    allWordsCount = len(allWords)

    result = []
    for i in progressbar(range(len(test_data)), "Processing Test Data: "):
        message = test_data[i]
        result.append(
            classify(
                message,
                ProbSpam,
                ProbNotSpam,
                spamWordCounts,
                spamWordCount,
                realWordCounts,
                realWordCount,
                allWords,
                allWordsCount,
            )
        )

    # print(result)
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

    naiveBayes(training_data, test_data, training_labels, test_labels)
