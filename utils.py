import copy
import cv2
import os
from PIL import Image
import math
from random import randint
import numpy as np
from keras.datasets import mnist


def insert_noise(noise_dir :str, output_dir :str, dataset, len_noise:int, saveImages:bool):

    datasets = np.empty((21,len(dataset),28,28))
    for x in range(0,101):
        if x % 5 == 0:
            opacity = x / 100
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + output_dir)

            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            imagelist = np.empty((len(dataset),28,28))
            factor = int(math.ceil(opacity * 255))
            for item in range(len(dataset)):
                picture = Image.fromarray(dataset[item])

                overlay_path = os.path.join(os.path.dirname(__file__), '../overlays/' + str(noise_dir) + '/' + str(randint(0,len_noise - 1)))
                overlay = Image.open(overlay_path)
                temp = picture.copy()
                pixeldata = list(overlay.getdata())
                overlay_copy = Image.new(overlay.mode, overlay.size)

                for i, pixel in enumerate(pixeldata):
                    if pixeldata[i][3] > 0:
                        pixeldata[i] = (255, 255, 255, factor)

                overlay_copy.putdata(pixeldata)

                temp.paste(overlay_copy, (0, 0), mask=overlay_copy)
                pix = np.array(temp)
                imagelist[item] = pix

                if saveImages:
                    base_dir_extended = base_dir + "/op" + str(opacity) + "/"
                    if not  os.path.exists(base_dir_extended):
                        os.makedirs(base_dir_extended)
                    temp.save(base_dir_extended + "/ " + str(item) + ".png")
            np.savez_compressed(base_dir + "/op" + str(opacity), imagelist)


def insert_vertical_bars(opacity: float, dataset: np.ndarray):
    dataset_copy = copy.deepcopy(dataset)

    #loop over all the columns and all the rows, if the row number is a multiple of 4, we add the noise by comparing the opacity to the number itself.
    for i in range(len(dataset_copy)):
        for j in range(len(dataset_copy[i])):
            if j % 4 == 0:
                for k in range(len(dataset_copy[i][j])):
                    if dataset_copy[i][j][k][0] < opacity:
                        dataset_copy[i][j][k] = np.array([opacity])
    return dataset_copy


def insert_random_noise(opacity, dataset: np.ndarray):
    dataset_copy = copy.deepcopy(dataset)

    #loop over all pixels and add noise by comparing it to the number itself.
    for i in range(len(dataset_copy)):
        for j in range(len(dataset_copy[i])):
            for k in range(len(dataset_copy[i][j])):
                if dataset_copy[i][j][k][0] < (opacity / 4):
                    dataset_copy[i][j][k] = np.array([opacity / 4])
    return dataset_copy

def distance(original: np.ndarray, modified: np.ndarray):

    #loop over all pixels in the new and original image, calculate the absolute difference and normalize the total.
    total = 0
    for i in range(len(original)):
        for j in range(len(original[i])):
            for k in range(len(original[i][j])):
                total += abs(original[i][j][k] - modified[i][j][k])

    factor = (len(original) * len(original[i]) * len(original[i][j]))
    print("factor : ", factor)
    return total / factor


def show_result(image):

    cv2.imshow("Digit", (image[0] * 255).astype("uint8"))
    cv2.waitKey(0)

def createnumbernoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_noise("numbers", "number_noise", x_test, 10, False)

def createbarnoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_noise("bars", "bar_noise", x_test, 1, False)

createbarnoise()

