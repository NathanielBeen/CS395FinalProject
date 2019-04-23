#imports
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from random import shuffle
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument

from sklearn.preprocessing import scale
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class LabeledData:
    def __init__(self, data, label):
        self.data = data
        self.label = label


# this will likely change depending on the method we want to use
# updated version labels each review so that we can shuffle the data
def preProcessData(data, small_stop_words=False, stop_words=False, stem=False, lemma=False):
    # define the HTML tags and punctuation we want to replace
    PUNC = re.compile("[.;:!\'?,\"()\[\]]")
    TAGS = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # replace punctuation as is, tags with a space
    data = [PUNC.sub("", line.lower()) for line in data]
    data = [TAGS.sub(" ", line.lower()) for line in data]

    if small_stop_words:
        to_remove = ['in','of','at','a','the']
        data = [' '.join([word for word in review.split() if word not in to_remove]) for review in tqdm(data)]

    if stop_words:
        to_remove = stopwords.words('english')
        data = [' '.join([word for word in review.split() if word not in to_remove]) for review in tqdm(data)]

    if stem:
        stemmer = PorterStemmer()
        data = [' '.join([stemmer.stem(word) for word in review.split()]) for review in tqdm(data)]

    if lemma:
        lemmatizer = WordNetLemmatizer()
        data = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in tqdm(data)]

    return data


def shuffleAndSeperateData(data):
    # convert to LabeledData so we dont lose labels when shuffled
    # the first half is positive and the last half is negative
    labeled_data = [LabeledData(data[i], (1 if i < (len(data) / 2) else 0)) for i in range(len(data))]

    # shuffle data
    shuffle(labeled_data)
    # seperate the data into x and y components
    x_data = [review.data for review in labeled_data]
    y_data = [review.label for review in labeled_data]
    return x_data, y_data


# this will also change depending on methodology
def processData(train, test, type, n_grams=None):
    if type == 'Count':
        return processCountVector(train, test, n_grams)
    elif type == 'Tfidf':
        return processTfIdfVector(train, test, n_grams)
    else:
        return processWord2Vec(train, test, type)

def processCountVector(train, test, n_grams=None):
    if n_grams is None:
        vectorizer = CountVectorizer(binary=False, max_features=5000)
    else:
        vectorizer = CountVectorizer(binary=False, max_features=5000, ngram_range=(1, n_grams))
    vectorizer.fit(train)
    return vectorizer.transform(train), vectorizer.transform(test), 5000

def processTfIdfVector(train, test, n_grams=None):
    if n_grams is None:
        vectorizer = TfidfVectorizer(max_features=5000)
    else:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, n_grams))
    vectorizer.fit(train)
    return vectorizer.transform(train), vectorizer.transform(test), 5000

def processWord2Vec(train, test, model_type):

    # turns each review into a TaggedDocument with a words attribute and a tags attribute
    # (this is needed because Doc2Vec requires a TaggedDocument input)
    def convertToTaggedDocument(data, label):
        result = []
        prefix = label
        for i in range(len(data)):
            result.append(TaggedDocument(words=data[i], tags=[prefix + '_%s' % i]))
        return result

    # takes the frequecy from the tfidf and combines it with the associations found in the word2vec
    def buildWordVector(words, size, model, tfidf):
        vector = np.zeros(size).reshape((1, size))
        count = 0
        for word in words:
            try:
                # add the associations found in word2vec scaled by the word's frequency
                vector += model[word].reshape((1, size)) * tfidf[word]
                count += 1
            except KeyError:
                continue
        if count != 0:
            # scale the vector by diving by total number of words in the revidw
            vector /= count
        return vector

    train_data = convertToTaggedDocument([[word for word in word_tokenize(review)] for review in tqdm(train)], 'Train')
    test_data = convertToTaggedDocument([[word for word in word_tokenize(review)] for review in tqdm(test)], 'Test')

    # create the desired word2vec model
    cores = multiprocessing.cpu_count()
    model = createWord2Vec(model_type, cores, train_data)
    print('finished creating Word2Vec \n')

    # get the relative frequency of words so we can weight the above word2vec matrix for each review
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in tqdm(train_data)])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('created Tfidf \n')

    # "scale" standardizes the data
    train_vec = scale(np.concatenate([buildWordVector(z.words, 1000, model, tfidf) for z in tqdm(train_data)]))
    test_vec = scale(np.concatenate([buildWordVector(z.words, 1000, model, tfidf) for z in tqdm(test_data)]))
    return train_vec, test_vec, 1000

def createWord2Vec(type, cores, train):
    if type == 'Word2Vec':
        model = Word2Vec(size=1000, min_count=10, window=5)
        model.build_vocab([x.words for x in tqdm(train)])
        model.train([x.words for x in tqdm(train)], total_examples=len(train), epochs=10)
        return model
    elif type == 'DBOW':
        model = Doc2Vec(dm=0, size=1000, window=5, negative=5, min_count=5, workers=cores, alpha=0.065, min_alpha=0.065)
        model.build_vocab([x for x in tqdm(train)])
        model.train([x for x in tqdm(train)], total_examples=len(train), epochs=10)
        return model
    elif type == 'DMC':
        model = Doc2Vec(dm=1, dm_concat=1, size=1000, window=5, negative=5, min_count=5, workers=cores, alpha=0.065, min_alpha=0.065)
        model.build_vocab([x for x in tqdm(train)])
        model.train([x for x in tqdm(train)], total_examples=len(train), epochs=10)
        return model
    else:
        model = Doc2Vec(dm=1, dm_mean=1, size=1000, window=5, negative=5, min_count=5, workers=cores, alpha=0.065, min_alpha=0.065)
        model.build_vocab([x for x in tqdm(train)])
        model.train([x for x in tqdm(train)], total_examples=len(train), epochs=10)
        return model


def buildModel(activation, loss, input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model

# different activation functions
activations = ['relu', 'tanh', 'sigmoid']
# different loss functions
losses = ['binary_crossentropy', 'mean_absolute_error', 'mean_squared_error']

for i in range(3):
    # read in collected data
    training_data = [line.strip() for line in open('./movie_data/full_train.txt')]
    testing_data = [line.strip() for line in open('./movie_data/full_test.txt')]

    # strip html and punctuation and convert to LabeledData
    pp_training = preProcessData(training_data, small_stop_words=True)
    pp_testing = preProcessData(testing_data, small_stop_words=True)

    # seperate the data and randomize it
    x_training, y_training = shuffleAndSeperateData(pp_training)
    x_testing, y_testing = shuffleAndSeperateData(pp_testing)

    # get the word vectors for the data
    x_training, x_testing, dimensions = processData(x_training, x_testing, 'Tfidf', n_grams=2)

    model = buildModel('relu', 'binary_crossentropy', dimensions)
    model.fit(x_training, y_training, epochs=5, validation_data=(x_testing, y_testing), batch_size=150, verbose=2)
    loss, accuracy = model.evaluate(x_testing, y_testing, verbose=False)
    print('Loss: '+str(loss))
    print('\n')
    print('Accuracy: ' + str(accuracy))
    print('\n')