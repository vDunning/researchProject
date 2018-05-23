from lenet import LeNet
import keras
import argparse
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("[INFO] compiling model..")

opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=num_classes, weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

if args["load_model"] < 0:
    print("[INFO] training ...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)


print("[INFO] evaluating ...")


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

def insert_vertical_bars(opacity: float, dataset: np.ndarray):
    dataset_copy = copy.deepcopy(dataset)
    for i in range(len(dataset_copy)):
        for j in range(len(dataset_copy[i])):
            if j % 4 == 0:
                for k in range(len(dataset_copy[i][j])):
                    if dataset_copy[i][j][k][0] < opacity:
                        dataset_copy[i][j][k] = np.array([opacity])
    return dataset_copy

xvalues = []
yvalues = []

for i in range(0, 10):
    factor = i / 10
    if i % 1 == 0:
        next_x_test = insert_vertical_bars(factor, x_test)
        (vertbarsloss, vertbarsaccuracy) = model.evaluate(next_x_test, y_test, batch_size=batch_size, verbose=1)
        print("Opacity : ", factor)
        print("[INFO] accuracy with horizontal bars : {:.2f}%".format(vertbarsaccuracy * 100))
        cv2.imshow("Digit", (next_x_test[0] * 255).astype("uint8"))
        cv2.waitKey(0)
        xvalues.append(factor)
        yvalues.append(vertbarsaccuracy)

plt.plot(xvalues, yvalues)
plt.show()

if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)
