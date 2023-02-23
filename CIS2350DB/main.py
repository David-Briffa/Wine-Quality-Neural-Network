from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as mpl

# used for plotting graphs
history = History()

# loading randomized dataset
data = pd.read_csv('winequality-red.csv', delimiter=';', header=0)
# encoding the quality column since it is categorical data
encoder = OneHotEncoder(handle_unknown='ignore')
dataframe = pd.DataFrame(data['quality'])
encoded_dataframe = pd.DataFrame(encoder.fit_transform(dataframe).toarray())
final_dataframe = dataframe.join(encoded_dataframe)

# dropped quality keyword as everything has been encoded
final_dataframe.drop('quality', axis=1, inplace=True)

# print(final_dataframe.head()) testing purposes
# print(np.unique(data['quality'])) # checking unique quality values
X = data.iloc[:, 11]
Y = final_dataframe
# print(Y.shape) confirming that dataframe worked

# dividing the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# checking shape of data post-splitting for debugging purposes
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

network = Sequential([
    Dense(11),                      # input layer
    LeakyReLU(alpha=0.05),          # Leaky ReLu has to be hardcoded
    Dropout(0.2),                   # dropout helps with avoiding overfitting
    Dense(11),                      # hidden layer
    LeakyReLU(alpha=0.05),
    Dropout(0.2),
    Dense(6, activation='sigmoid')  # output layer with 6 neurons as there are 6 unique quality levels
])

network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# different epoch settings
# history = network.fit(x_train, y_train, validation_split=0.2, epochs=10)
# history = network.fit(x_train, y_train, validation_split=0.2, epochs=80)
history = network.fit(x_train, y_train, validation_split=0.2, epochs=300)
# history = network.fit(x_train, y_train, validation_split=0.2, epochs=1000)

print()
print("The Below is the accuracy of the Neural Network using the testing data")
_, testingAccuracy = network.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (testingAccuracy*100))

# graph to visualize increase in training accuracy in relation to loss reduction
mpl.figure(figsize=[8, 6])
mpl.plot(history.history['accuracy'], 'b', linewidth=3.0)
mpl.plot(history.history['loss'], 'r', linewidth=3.0)
mpl.legend(['Accuracy', 'Loss'],fontsize=18)
mpl.xlabel('Epochs ', fontsize=16)
mpl.ylabel('Accuracy', fontsize=16)
mpl.title('Accuracy Curves', fontsize=16)
mpl.show()

# graph to visualize how accurately the validation data is classified
mpl.figure(figsize=[8, 6])
mpl.plot(history.history['accuracy'], 'b', linewidth=3.0)
mpl.plot(history.history['val_accuracy'], 'r', linewidth=3.0)
mpl.legend(['Accuracy', 'Validation Accuracy'], fontsize=18)
mpl.xlabel('Epochs ', fontsize=16)
mpl.ylabel('Accuracy', fontsize=16)
mpl.title('Accuracy vs Validation', fontsize=16)
mpl.show()