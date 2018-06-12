from lenet import LeNet
import keras
import argparse
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
import os
import matplotlib.pyplot as plt
import utils
import pandas as pd
import seaborn as sb
import numpy as np


print("TEST")
batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Modify the order of the inputs, channel first or channel last
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#set the values in the pixels as floats and normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Create 10 categories (0..9) for the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("[INFO] compiling model..")

#we use stochastic gradient descent to train our network with a learning rate of 0.01
opt = SGD(lr=0.01)

#initialize the model with the given parameters.
model = LeNet.build(width=28, height=28, depth=1, classes=num_classes, weightsPath=args["weights"] if args["load_model"] > 0 else None)

#we use categorical_crossentropy as our loss function
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

if args["load_model"] < 0:
    print("[INFO] training ...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)


print("[INFO] evaluating ...")


def calc_acc_multopacities(dataset_dir: str):

    xvalues = []
    yvalues = []

    for x in range(0, 101):
        if x % 5 == 0:
            opacity = x / 100
            print("TESTING OPACITY:", opacity)
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + dataset_dir + "/")
            dataset = np.load(base_dir + 'op' + str(opacity) + ".npz")
            testset = dataset['arr_0'].reshape((10000,28,28,1))
            (loss, accuracy) = model.evaluate(x_test, y_test, batch_size = batch_size, verbose=1)
            print("[INFO] accuracy with noise : {:.2f}%".format(accuracy * 100))
            distance = utils.distance(x_test, testset)
            print("opacity : " + str(opacity) + ", distance : " + str(distance))
            xvalues.append(distance)
            yvalues.append(accuracy)

    plt.plot(xvalues, yvalues)
    plt.show()


def calc_accuracy_multopacities():

    xvalues = []
    yvaluesrand = []
    yvaluesbars = []

    #Loop over multiple opacities
    for i in range(0, 10):
        factor = i / 10

        #create the next data sets for random noise and horizontal bars.
        next_x_test_rand = utils.insert_random_noise(factor, x_test)
        next_x_test_bars = utils.insert_vertical_bars(factor, x_test)

        #evaluate the vertical bars and the random noise loss and accuracy
        (vertbarsloss, vertbarsaccuracy) = model.evaluate(next_x_test_bars, y_test, batch_size=batch_size, verbose=1)
        (randloss, randaccuracy) = model.evaluate(next_x_test_rand, y_test, batch_size=batch_size, verbose=1)
        print("[INFO] accuracy with horizontal bars : {:.2f}%".format(vertbarsaccuracy * 100))
        print("[INFO] accuracy with random noise : {:.2f}%".format(randaccuracy * 100))

        #calculate the distance of the dataset and plot it
        distance = utils.distance(x_test, next_x_test_bars)
        print("opacity : " + str(factor) + ", distance : " + str(distance))
        xvalues.append(distance)
        yvaluesrand.append(randaccuracy)
        yvaluesbars.append(vertbarsaccuracy)

    plt.plot(xvalues, yvaluesrand)
    plt.plot(xvalues, yvaluesbars)
    plt.show()

def calc_confusion_matrix():

    matrix = [[0] * 10 for x in range(0,10)]
    x_test_bars = utils.insert_horizontal_bars(1, x_test)
    prediction = model.predict_classes(x_test_bars)
    for i in range(len(x_test)):
        matrix[y_test[i].tolist().index(1)][prediction[i]] += 1

    df_cm = pd.DataFrame(matrix, index = [i for i in range(0,10)], columns=[i for i in range(0,10)])
    sb.set(font_scale=1.4)
    sb.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted class")
    plt.ylabel("actual class")
    plt.show()
    print(matrix)

if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

calc_acc_multopacities("number_noise")

# for i in np.random.choice(np.arange(0, len(x_test_vertbars)), size=(2,)):
#
#     imagemod = (x_test_vertbars[i] * 255).astype("uint8")
#     imagereal = (x_test[i] * 255).astype("uint8")
#
#     print("*********************")
#     print(type(x_test[i][0][0][0]))
#     print("======================")
#     print(type(x_test_vertbars[i][0][0][0]))
#     cv2.imshow("Digit", imagemod)
#     cv2.waitKey(0)
#     cv2.imshow("Digit", imagereal)
#     cv2.waitKey(0)