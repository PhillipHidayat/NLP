# models.py

from sentiment_data import *
from sentiment_data import List
from utils import *

from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import string
import random

stop_words = set(stopwords.words('english'))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.vocab = indexer
        self.weight = []
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        sentence = lower(sentence)
        for word in sentence:
            # if add_to_indexer true
            if(add_to_indexer):
                # add new words to vocab
                if(not self.vocab.contains(word)):
                   index = self.vocab.add_and_get_index(word)
                   self.weight.insert(index, 0)
            else:
                if(not self.vocab.contains(word)):
                    # remove word from sentence
                    sentence.remove(word)
        
        # get counts of words that is in vocab
        self.feature_vector = Counter(sentence)

        return self.feature_vector


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.feature_vector = Counter()
        self.vocab = indexer
        self.weight = []
        
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # create bigram words list
        bigram_words = []
        for i in range(len(sentence) - 1):
            bigram_words.append(sentence[i] + " " + sentence[i+1])

        for word in bigram_words:
            # if add_to_indexer true
            if(add_to_indexer):
                # add new words to vocab
                if(not self.vocab.contains(word)):
                   index = self.vocab.add_and_get_index(word)
                   self.weight.insert(index, 0)
            else:
                if(not self.vocab.contains(word)):
                    bigram_words.remove(word)
        
        # get counts of words that is in vocab
        self.feature_vector = Counter(bigram_words)

        return self.feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.feature_vector = Counter()
        self.vocab = indexer
        self.weight = []
        
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        # create 3-gram words list
        trigram_words = []
        for i in range(len(sentence) - 2):
            trigram_words.append(sentence[i] + " " + sentence[i+1] + " " + sentence[i+2])

        for word in trigram_words:
            # if add_to_indexer true
            if(add_to_indexer):
                # add new words to vocab
                if(not self.vocab.contains(word)):
                   index = self.vocab.add_and_get_index(word)
                   self.weight.insert(index, 0)
            else:
                if(not self.vocab.contains(word)):
                    trigram_words.remove(word)
        
        # get counts of words that is in vocab
        self.feature_vector = Counter(trigram_words)

        return self.feature_vector

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence)
        dot_product = dot_product_list(self.feat_extractor.weight, feature_vector, self.feat_extractor.vocab)
        return 1 if dot_product > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence)
        dot_product = dot_product_list(self.feat_extractor.weight, feature_vector, self.feat_extractor.vocab)
        prob = LR_pos_prob(dot_product)
        return 1 if prob > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    percepton = PerceptronClassifier(feat_extractor)
    epoch = 60
    multiplier = 0.9

    for i in range(epoch):
        for example in train_exs:
            # print("vocab is ", feat_extractor.vocab)
            words = example.words
            sentence = filter_words(words, stop_words)
            feature_vector = feat_extractor.extract_features(sentence, True)
            prediction = percepton.predict(sentence)
            # update weight
            if (example.label == 1 and prediction != 1):                
                for feature in feature_vector.keys():
                    index = feat_extractor.vocab.index_of(feature)
                    feat_extractor.weight[index] += multiplier * feature_vector[feature]
            elif (example.label == 0 and prediction != 0):            
                for feature in feature_vector.keys():
                    index = feat_extractor.vocab.index_of(feature)
                    feat_extractor.weight[index] -= multiplier * feature_vector[feature]                 

    return percepton


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    learning_rate = 0.05
    epochs = 10
   
    
    for i in range(epochs):
        random.shuffle(train_exs)
        for example in train_exs:
            y = example.label
            sentence = example.words
            feature_vector = feat_extractor.extract_features(sentence, True)

            dot_product = dot_product_list(feat_extractor.weight, feature_vector, feat_extractor.vocab)
            prob = LR_pos_prob(dot_product)
            
            for feature in feature_vector.keys():
                index = feat_extractor.vocab.index_of(feature)
                lf = (prob - y) * feature_vector[feature]
                feat_extractor.weight[index] -= learning_rate * lf
                if(feat_extractor.weight[index] > 100):
                    print(feat_extractor.weight[index], index)

    return LogisticRegressionClassifier(feat_extractor)



def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

def filter_words(words, stop_words):
    """
    remove stopwords and punctuation and return remaining list of words
    """
    filtered_word = []
    for word in words:
        if(word.lower() not in stop_words and not any(p in word for p in list(string.punctuation))):
            filtered_word.append(word)
    return filtered_word

def dot_product_list(weight: List, feature_vector: Counter, vocab: Indexer):
    dot_product = 0
    for feature in feature_vector.keys():
        index = vocab.index_of(feature)
        dot_product += weight[index] * feature_vector[feature]  

    return dot_product

def LR_pos_prob(z):
    exp = np.exp(z)
    prob = exp / (1 + exp)
    return prob

def lower(words):
    list = []
    for word in words:
        list.append(word.lower())
    return list
