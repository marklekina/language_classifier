#!/bin/python
#
# classifier.py - extracts useful features from mixed language dataset;
#                 train a Naive Bayes classifier model from extracted features;
#                 assess the performance of the model (accuracy score, most informative features, classification report);
#                 display a histogram distribution of the predicted language groupings by mean sentence length.
#
# Mark Lekina Rorat, July, Sept, Oct 2021


import random

import pandas as pd

# import libraries
from itertools import chain
from nltk import BigramAssocMeasures, BigramCollocationFinder, word_tokenize, NaiveBayesClassifier, classify, \
    sent_tokenize
from nltk.classify import apply_features
from nltk.metrics import ConfusionMatrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# function to extract features from text, i.e.
# (1) construct a bag of words with both unigrams and bigrams (https://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/) and
# (2) get mean sentence length

def get_features(text, score_fn=BigramAssocMeasures.chi_sq, num=100):
    # tokenize text
    tokens = word_tokenize(text)

    # find and rank bigram collocations from text or empty list if n/a
    try:
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigrams = bigram_finder.nbest(score_fn, num)
    except ZeroDivisionError:
        bigrams = []

    # generate unigrams from text
    unigrams = []
    for token in tokens:
        pair = tuple([token])
        unigrams.append(pair)
    # return combined bag of words
    features = [(ngram, 1.0) for ngram in chain(bigrams, unigrams)]
    features.append((('mean_sentence_length',), mean_sentence_length(text)))
    return dict(features)


# function to plot histograms
def plot_histogram(a_list, label, bins=10):
    plt.title(label)
    plt.hist(a_list, bins=bins)
    plt.show()


# function to find average sentence lengths
def mean_sentence_length(text):
    sentences = sent_tokenize(text)
    count = 0
    for sentence in sentences:
        count += len(sentence)
    mean = count / len(sentences)
    return mean


# function to load data, train and test the classifier model, and make predictions from chat data
def language_classifier(dataset_file, threshold=0.25):
    # load data
    dev_data = pd.read_csv(dataset_file)
    labelled_text = [(row.text, row.label) for index, row in dev_data.iterrows()]
    random.shuffle(labelled_text)

    # split train and test sets and generate feature sets
    cutoff = int(threshold * len(labelled_text))
    train_set = apply_features(get_features, labelled_text[cutoff:])
    test_set = apply_features(get_features, labelled_text[:cutoff])

    # train classifier model
    classifier = NaiveBayesClassifier.train(train_set)

    # print accuracy of the model and the most informative features
    print('Classifier accuracy: ', classify.accuracy(classifier, test_set), '\n')
    classifier.show_most_informative_features(20)

    # generate and print classification report
    gold_labels = [pair[1] for pair in labelled_text[:cutoff]]
    predicted_labels = [classifier.classify(get_features(pair[0])) for pair in labelled_text[:cutoff]]
    print('\n', classification_report(gold_labels, predicted_labels))
    print(ConfusionMatrix(gold_labels, predicted_labels))

    # display histograms of sentence lengths
    text_label_pairs = [[pair[0], classifier.classify(get_features(pair[0]))] for pair in labelled_text[:cutoff]]
    sheng_pairs, swahili_pairs, english_pairs = [], [], []
    for pair in text_label_pairs:
        if pair[1] == 'sheng':
            sheng_pairs.append(mean_sentence_length(pair[0]))
        elif pair[1] == 'swahili':
            swahili_pairs.append(mean_sentence_length(pair[0]))
        elif pair[1] == 'english':
            english_pairs.append(mean_sentence_length(pair[0]))

    plot_histogram(sheng_pairs, 'sheng')
    plot_histogram(swahili_pairs, 'swahili')
    plot_histogram(english_pairs, 'english')


if __name__ == "__main__":
    language_classifier('data/language_dataset.csv')
