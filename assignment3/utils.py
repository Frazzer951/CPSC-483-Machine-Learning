import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def load_data():
    return pd.read_csv("emails.csv")


def filter_punctuation(line):
    punc = string.punctuation
    for char in punc:
        line = line.replace(char, "")
    return line


def filter_stopwords(line):
    sw = stopwords.words("english")
    words = line.split()
    new_line = ""
    for word in words:
        word = word.lower()
        if word not in sw:
            new_line = new_line + " " + word

    return new_line.strip()


def preprocess(data: pd.DataFrame):
    data["text"] = data["text"].apply(filter_punctuation)
    data["text"] = data["text"].apply(filter_stopwords)

    return data


def split_data(data, tt_ratio=0.50):  # By default have a split of 50% training and 50% testing
    feats = data["text"]
    labels = data["spam"]
    training_data, test_data, training_labels, test_labels = train_test_split(
        feats, labels, test_size=tt_ratio, random_state=42
    )
    return (
        training_data.to_numpy(),
        test_data.to_numpy(),
        training_labels.to_numpy(dtype=np.bool_),
        test_labels.to_numpy(dtype=np.bool_),
    )
