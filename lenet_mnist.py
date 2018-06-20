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
import json

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


def calc_acc_multopacities(dataset_dir: str, calcdistance):
    xvalues = []
    yvalues = []
    for x in range(0, 101):
        if x % 5 == 0:
            opacity = x / 100
            print("TESTING OPACITY:", opacity)
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + dataset_dir + "/")
            print(base_dir + 'op' + str(opacity) + ".npz")
            dataset = np.load(base_dir + 'op' + str(opacity) + ".npz")
            testset = dataset['arr_0'].reshape((10000,28,28,1))
            testset = testset.astype('float32')
            testset /= 255
            (loss, accuracy) = model.evaluate(testset, y_test, batch_size = batch_size, verbose=1)
            print("[INFO] accuracy with noise : {:.2f}%".format(accuracy * 100))
            if calcdistance:
                distance = utils.distance(x_test, testset)
                xvalues.append(distance)
                print("opacity : " + str(opacity) + ", distance : " + str(distance))
            yvalues.append(accuracy)
    return (xvalues, yvalues)

def calc_confidence(dataset_dir, calcdistance):
    xvalues = []
    yvalues = []
    for x in range(0, 101):
        if x % 5 == 0:
            opacity = x / 100
            print("TESTING OPACITY:", opacity)
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + dataset_dir + "/")
            print(base_dir + 'op' + str(opacity) + ".npz")
            dataset = np.load(base_dir + 'op' + str(opacity) + ".npz")
            testset = dataset['arr_0'].reshape((10000, 28, 28, 1))
            testset = testset.astype('float32')
            testset /= 255
            predictions = model.predict(testset)
            totalconfidence = 0
            for i in range(0, len(predictions)):
                confidence = np.amax(predictions[i])
                if predictions[i].tolist().index(confidence) == y_test[i].tolist().index(1):
                    totalconfidence += confidence
            totalconfidence = totalconfidence / len(predictions)
            if calcdistance:
                distance = utils.distance(x_test, testset)
                xvalues.append(distance)
            yvalues.append(totalconfidence)
            print("Distance: " + str(distance) + " Confidence: " + str(totalconfidence))
    return (xvalues, yvalues)


def calc_confusion_matrix(dir):

    matrix = [[0] * 10 for x in range(0,10)]
    opacity = 0.8
    x_test_noise = np.load("../datasets/" + dir + "/op" + str(opacity) + ".npz")['arr_0']
    x_test_noise = x_test_noise.astype('float32')
    x_test_noise /= 255
    prediction = model.predict_classes(x_test_noise)
    for i in range(len(x_test)):
        matrix[y_test[i].tolist().index(1)][prediction[i]] += 1
    df_cm = pd.DataFrame(matrix, index = [i for i in range(0,10)], columns=[i for i in range(0,10)])
    sb.set(font_scale=1.4)
    sb.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted class")
    plt.ylabel("actual class")
    plt.show()


if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)


def calc_number_distortion(opacity):
    base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + 'number_noise' + "/")
    dataset = np.load(base_dir + 'op' + str(opacity) + ".npz")
    testset = dataset['arr_0'].reshape((10000,28,28,1))
    testset = testset.astype('float32')
    testset /= 255
    prediction = model.predict_classes(testset)
    addednoise = np.load(base_dir + 'addednoise' + str(opacity) + ".npz")
    addednoise = addednoise['arr_0']
    correct = 0
    asnoise = 0
    asothernumber = 0
    aseight = 0
    for i in range(len(y_test)):
        actual = y_test[i].tolist().index(1)
        predicted = prediction[i]
        noise = addednoise[i]
        if actual == predicted:
            correct += 1
        elif predicted == noise:
            if predicted == 8:
                aseight += 1
            asnoise += 1
        else:
            asothernumber += 1
            if predicted == 8:
                aseight += 1
    print("CORRECTLY PREDICTED: ", correct / len(y_test))
    print("PREDICTED AS NOISE: ", asnoise / len(y_test))
    print("PREDICTED AS OTHER NUMBER: ", asothernumber / len(y_test))
    print("PREDICTED AS EIGHT: ", aseight / len(y_test))
    return correct / len(y_test) * 100, asnoise / len(y_test) * 100, asothernumber / len(y_test) * 100

def calculate_prediction_spread_numbernoise():
    ycorrect = []
    yasnoise = []
    yasothernumber = []
    xvalues = []
    for x in range(0, 101):
        if x % 5 == 0:
            opacity = x / 100
            xvalues.append(opacity)
            correctpredicted, predictedasnoise, predictedasothernumber = calc_number_distortion(opacity)
            ycorrect.append(correctpredicted)
            yasnoise.append(predictedasnoise)
            yasothernumber.append(predictedasothernumber)
    plt.plot(xvalues, ycorrect, label='% of images correctly predicted')
    plt.plot(xvalues, yasnoise, label='% of images predicted as added noise')
    plt.plot(xvalues, yasothernumber, label='% of images predicted as arbitrary number')
    plt.legend()
    plt.xlabel('opacity')
    plt.ylabel('accuracy')
    plt.show()

def calculate_accuracies():
    xshape, yshape = calc_acc_multopacities('random_shapes/20shapes', True)
    xrand, yrand = calc_acc_multopacities('random_pixels', True)
    xnum, ynum = calc_acc_multopacities('number_noise', True)
    xbar, ybar = calc_acc_multopacities('bar_noise', True)

    results = dict()
    results['Shapes'] = (xshape, yshape)
    results['Random'] = (xrand, yrand)
    results['Numbers'] = (xnum, ynum)
    results['Bars'] = (xbar, ybar)
    # with open('../results/accuracies.json', 'w') as file:
    #     json.dump(results, file)

    plt.plot(xshape,yshape, label = 'Shapes')
    plt.plot(xrand,yrand, label = 'Random Noise')
    plt.plot(xnum,ynum, label = 'Other numbers')
    plt.plot(xbar,ybar, label = 'Horizontal bars')
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Accuracy')
    plt.show()

def calculate_confidence():
    xshape, yshape = calc_confidence('random_shapes/20shapes', True)
    xrand, yrand = calc_confidence('random_pixels', True)
    xnum, ynum = calc_confidence('number_noise', True)
    xbar, ybar = calc_confidence('bar_noise', True)

    results = dict()
    results['Shapes'] = (xshape, yshape)
    results['Random'] = (xrand, yrand)
    results['Numbers'] = (xnum, ynum)
    results['Bars'] = (xbar, ybar)
    # with open('../results/accuracies.json', 'w') as file:
    #     json.dump(results, file)

    plt.plot(xshape,yshape, label = 'Test')
    plt.plot(xrand,yrand, label = 'Random Noise')
    plt.plot(xnum,ynum, label = 'Other numbers')
    plt.plot(xbar,ybar, label = 'Horizontal bars')
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Confidence')
    plt.show()

calculate_confidence()
# plt.plot(xbars, ybars, label='horizontal bars')
# plt.plot(xrand, yrand, label='random pixels')
# plt.plot(xnumbers, ynumbers, label='other numbers')
# plt.legend()
# plt.xlabel('distance')
# plt.ylabel('accuracy')
# plt.show()
# for x in range(0, 180):
#     if x % 30 == 0:
#         distance = False
#         if x == 0:
#             distance = True
#         (xvalues,yvalues) = calc_acc_multopacities('rotated_bars/rotation_' + str(x), distance)
#         plt.plot(xvalues, yvalues, label='rotation: ' + str(x))
# plt.legend()
# plt.xlabel('distance')
# plt.ylabel('accuracy')
# plt.show()

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