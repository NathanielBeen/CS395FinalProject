#imports
import re
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def createLabels(data_length):
    # the first half of the data is positive and the last half is negative
    return [1 if i < (data_length / 2) else 0 for i in range(data_length)]


# this will likely change depending on the method we want to use
def preProcessData(data):
    # define the HTML tags and punctuation we want to replace
    PUNC = re.compile("[.;:!\'?,\"()\[\]]")
    TAGS = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # replace punctuation as is, tags with a space
    data = [PUNC.sub("", line.lower()) for line in data]
    data = [TAGS.sub(" ", line.lower()) for line in data]
    return data

# this will also change depending on methodology
def processData(train, test):
    vectorizer = CountVectorizer(binary=True, max_features=5000)
    vectorizer.fit(train)
    return vectorizer.transform(train), vectorizer.transform(test)

# do we want different models or is it too much?
def buildModel(activation, loss):
    model = Sequential()
    model.add(Dense(128, input_dim=5000, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer = 'adam', metrics = ['accuracy'])
    return model

#different activation functions
activations = ['relu', 'tanh', 'sigmoid']
#different loss functions
losses = ['binary_crossentropy', 'mean_absolute_error', 'mean_squared_error']

#read in collected data
training_data = [line.strip() for line in open('./movie_data/full_train.txt')]
testing_data = [line.strip() for line in open('./movie_data/full_test.txt')]
y_training = createLabels(len(training_data))
y_testing = createLabels(len(testing_data))

pp_training = preProcessData(training_data)
pp_testing = preProcessData(testing_data)

x_training, x_testing = processData(pp_training, pp_testing)

for activation in activations:
    model = buildModel(activation, 'binary_crossentropy')

    model.fit(x_training, y_training, epochs=2, validation_data=(x_testing, y_testing), batch_size=150, verbose=2)
    accuracy = model.evaluate(x_testing, y_testing, verbose=False)[1]
    print('Accuracy: '+str(accuracy)+' using the '+activation+' activation function')
    print('\n')

for loss in losses:
    model = buildModel('relu', loss)

    model.fit(x_training, y_training, epochs=2, validation_data=(x_testing, y_testing), batch_size=150, verbose=2)
    accuracy = model.evaluate(x_testing, y_testing, verbose=False)[1]
    print('Accuracy: '+str(accuracy)+' using the '+loss+' loss function')
    print('\n')