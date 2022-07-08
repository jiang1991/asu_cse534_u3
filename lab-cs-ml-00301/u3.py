# -*- coding: utf-8 -*-


# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

# Splitting the dataset into the Training set and Test set (75% of data are used for training)
# reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split

# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is
# MinMaxScaler.
# referece: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import StandardScaler

# Encoding categorical data (convert letters/words in numbers)
# Reference: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# The following code work without warning in Python 3.6 or older. Newer versions suggest to use ColumnTransformer
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3])
X = onehotencoder.fit_transform(X).toarray()
'''
# The following code work Python 3.7 or newer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense

DataPath = "D:/python/Unit/lab-cs-ml-00301"


def run_sa():
    train_data = "/Sa/Training-a1-a3.csv"
    test_data = "/Sa/Testing-a2-a4.csv"
    return train_data, test_data


def run_sb():
    train_data = "/Sb/Training-a1-a2.csv"
    test_data = "/Sb/Testing-a1.csv"
    return train_data, test_data


def run_sc():
    train_data = "/Sc/Training-a1-a2.csv"
    test_data = "/Sc/Testing-a1-a2-a3.csv"
    return train_data, test_data


def run_fnn(train_data, test_data):
    ########################################
    # Part 1 - Data Pre-Processing
    #######################################

    # Batch Size
    BatchSize = 10
    # Epohe Size
    NumEpoch = 10

    # Import dataset.
    # Dataset is given in TraningData variable You can replace it with the file

    # train data
    train_dataset = pd.read_csv(DataPath + train_data, header=None)
    X_train = train_dataset.iloc[:, 0:-2].values
    train_label_column = train_dataset.iloc[:, -2].values
    y_train = []
    for i in range(len(train_label_column)):
        if train_label_column[i] == 'normal':
            y_train.append(0)
        else:
            y_train.append(1)

    # Convert ist to array
    y_train = np.array(y_train)

    # test data
    test_dataset = pd.read_csv(DataPath + test_data, header=None)
    X_test = test_dataset.iloc[:, 0:-2].values
    test_label_column = test_dataset.iloc[:, -2].values
    y_test = []
    for i in range(len(test_label_column)):
        if test_label_column[i] == 'normal':
            y_test.append(0)
        else:
            y_test.append(1)

    # Convert ist to array
    y_test = np.array(y_test)

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [1, 2, 3])],
        # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
        remainder='passthrough'  # Leave the rest of the columns untouched
    )

    # 归一化处理
    # 将训练数据集和测试数据集合并，做一个归一transform，然后使用这个transform来归一测试与训练数据集
    X_all = np.vstack((X_train, X_test))
    ct.fit(X_all)
    X_train = np.array(ct.transform(X_train), dtype=np.float)
    X_test = np.array(ct.transform(X_test), dtype=np.float)
    print(type(X_train))

    # X_train = np.array(ct.fit_transform(X_train), dtype=np.float)
    print(X_train.shape)

    # X_test = np.array(ct.fit_transform(X_test), dtype=np.float)
    print(X_test.shape)

    # ================== #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # ================== #

    # 归一化 --- 因为数据集
    sc = StandardScaler()
    X_all = np.vstack((X_train, X_test))
    sc.fit(X_all)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    # X_train = sc.fit_transform(X_train)  # Scaling to the range [0,1]
    # X_test = sc.fit_transform(X_test)

    ########################################
    # Part 2: Building FNN
    #######################################

    # Initialising the ANN
    # Reference: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    classifier = Sequential()

    # Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables rectified
    # linear unit activation function relu, reference:
    # https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=len(X_train[0])))

    # Adding the second hidden layer, 6 nodes
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer, 1 node,
    # sigmoid on the output layer is to ensure the network output is between 0 and 1
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN, Gradient descent algorithm “adam“, Reference:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ This loss is for a binary
    # classification problems and is defined in Keras as “binary_crossentropy“, Reference:
    # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    # Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
    # add verbose=0 to turn off the progress report during the training
    # To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
    classifierHistory = classifier.fit(X_train, y_train, batch_size=BatchSize, epochs=NumEpoch)

    # evaluate the keras model for the provided model and dataset
    loss, accuracy = classifier.evaluate(X_train, y_train)
    print('Print the loss and the accuracy of the model on the dataset')
    print('Loss [0,1]: %.4f' % loss, 'Accuracy [0,1]: %.4f' % accuracy)

    ########################################
    # Part 3 - Making predictions and evaluating the model
    #######################################

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.9)  # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
    # summarize the first 5 cases
    # for i in range(5):
    # print('%s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))

    # Making the Confusion Matrix
    # [TN, FP ]
    # [FN, TP ]
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print('Print the Confusion Matrix:')
    print('[ TN, FP ]')
    print('[ FN, TP ]=')
    print(cm)

    ########################################
    # Part 4 - Visualizing
    #######################################

    # Import matplot lib libraries for plotting the figures.
    import matplotlib.pyplot as plt

    # You can plot the accuracy
    print('Plot the accuracy')
    # Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
    # use the command python -c 'import keras; print(keras.__version__)' on MAC or Linux to check Keras' version
    plt.plot(classifierHistory.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('accuracy_sample.png')
    plt.show()

    # You can plot history for loss
    print('Plot the loss')
    plt.plot(classifierHistory.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('loss_sample.png')
    plt.show()


if __name__ == '__main__':
    # train, test = run_sa()
    train, test = run_sb()
    # train, test = run_sc()
    run_fnn(train, test)
