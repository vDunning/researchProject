import copy
import cv2
import os
from PIL import Image
import math
import random
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


def insert_noise(noise_dir :str, output_dir :str, dataset, len_noise:int, saveImages:bool, angle):

    datasets = np.empty((21,len(dataset),28,28))
    for x in range(0,101):
        if x % 5 == 0:
            opacity = x / 100
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + output_dir)

            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            imagelist = np.empty((len(dataset),28,28,1))
            factor = int(math.ceil(opacity * 255))
            addednoise = np.empty(len(dataset))
            for item in range(len(dataset)):
                picture = Image.fromarray(dataset[item])
                noisenumber = random.randint(0,len_noise - 1)
                addednoise[item] = noisenumber
                overlay_path = os.path.join(os.path.dirname(__file__), '../overlays/' + str(noise_dir) + '/' + str(noisenumber))
                overlay = Image.open(overlay_path)
                temp = picture.copy()
                pixeldata = list(overlay.getdata())
                overlay_copy = Image.new(overlay.mode, overlay.size)
                for i, pixel in enumerate(pixeldata):
                    if pixeldata[i][3] > 0:
                        pixeldata[i] = (255, 255, 255, factor)

                overlay_copy.putdata(pixeldata)
                overlay_copy = overlay_copy.rotate(angle, expand=False)
                temp.paste(overlay_copy, (0, 0), mask=overlay_copy)
                pix = np.array(temp)
                imagelist[item] = pix.reshape((28,28,1))

                if saveImages:
                    base_dir_extended = base_dir + "/op" + str(opacity) + "/"
                    if not  os.path.exists(base_dir_extended):
                        os.makedirs(base_dir_extended)
                    temp.save(base_dir_extended + "/ " + str(item) + ".png")
            np.savez_compressed(base_dir + "/op" + str(opacity), imagelist)
            np.savez(base_dir + "/addednoise" + str(opacity), addednoise)

def insert_shapes(dataset, saveImages):
    datasets = np.empty((21, len(dataset), 28, 28))
    for x in range(0, 101):
        if x % 5 == 0:
            opacity = x / 100
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + 'random_shapes/20shapes')

            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            imagelist = np.empty((len(dataset), 28, 28, 1))
            factor = int(math.ceil(opacity * 255))
            for item in range(len(dataset)):
                picture = Image.fromarray(dataset[item])
                temp = picture.copy()
                for i in range(0, 20):
                    noisenumber = random.randint(0, 1)
                    xpos = random.randint(0, 27)
                    ypos = random.randint(0, 27)
                    overlay_path = os.path.join(os.path.dirname(__file__),
                                                '../overlays/' + 'shapes' + '/' + str(noisenumber))
                    overlay = Image.open(overlay_path)
                    temp = temp.copy()
                    pixeldata = list(overlay.getdata())
                    overlay_copy = Image.new(overlay.mode, overlay.size)
                    for i, pixel in enumerate(pixeldata):
                        if pixeldata[i][3] > 0:
                            pixeldata[i] = (255, 255, 255, factor)
                    angle = random.randint(0, 360)
                    overlay_copy.putdata(pixeldata)
                    overlay_copy = overlay_copy.rotate(angle, expand=True)
                    temp.paste(overlay_copy, (xpos, ypos), mask=overlay_copy)
                pix = np.array(temp)
                imagelist[item] = pix.reshape((28, 28, 1))

                if saveImages:
                    base_dir_extended = base_dir + "/op" + str(opacity) + "/"
                    if not os.path.exists(base_dir_extended):
                        os.makedirs(base_dir_extended)
                    temp.save(base_dir_extended + "/ " + str(item) + ".png")
            np.savez_compressed(base_dir + "/op" + str(opacity), imagelist)


def remove_pixels(dataset, saveImages):
    print(len(dataset))
    for fraction in range(0,101):
        if fraction % 5 == 0:
            percentage = fraction / 100
        base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + 'removed_pixels')
        imagelist = [[[0 for x in range(28)] for y in range(28)] for z in range(len(dataset))]
        print(len(imagelist))
        dataset_copy = copy.deepcopy(dataset)
        for i in range(len(dataset_copy)):
            for j in range(len(dataset_copy[i])):
                for k in range(len(dataset_copy[i][j])):
                    if dataset_copy[i][j][k] > 0:
                        number = random.random()
                        #print("percentage: " + str(percentage) + "number: " + str(number))
                        if number < percentage:
                            dataset_copy[i][j][k] = np.array([0])
            if saveImages:
                base_dir_extended = base_dir + "/op" + str(percentage) + "/"
                if not os.path.exists(base_dir_extended):
                    os.makedirs(base_dir_extended)
                temp = Image.fromarray(dataset_copy[i])
                temp.save(base_dir_extended + "/ " + str(i) + ".png")
            imagelist[i] = np.array(temp).reshape((28,28,1))
        np.savez_compressed(base_dir + "/op" + str(percentage), imagelist)

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

def insert_random_pixels(dataset, saveImages):

    for x in range(0,101):
        if x % 5 == 0:
            opacity = x / 100
            base_dir = os.path.join(os.path.dirname(__file__), '../datasets/' + 'random_pixels')

            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            imagelist = [[[0 for x in range(28)] for y in range(28)] for z in range(len(dataset))]
            print(type(imagelist[0][0][0]))
            factor = int(math.ceil(opacity * 255))
            for item in range(len(dataset)):
                overlaypixels = np.zeros((28, 28))
                xcoordinates = [None] * 196
                ycoordinates = [None] * 196
                for j in range(0, 196):
                    xcoordinates[j] = random.randint(0, 27)
                    ycoordinates[j] = random.randint(0, 27)
                for i in range(0, len(xcoordinates)):
                    overlaypixels[xcoordinates[i]][ycoordinates[i]] = math.ceil(opacity * 255)
                overlaypixels = overlaypixels.astype('uint8')
                overlay = Image.fromarray(overlaypixels, 'L')
                temp = Image.fromarray(dataset[item]).copy()
                temp.paste(overlay, (0, 0), mask=overlay)
                pix = np.array(temp)
                imagelist[item] = pix.reshape((28, 28, 1))

                if saveImages:
                    base_dir_extended = base_dir + "/op" + str(opacity) + "/"
                    if not  os.path.exists(base_dir_extended):
                        os.makedirs(base_dir_extended)
                    temp.save(base_dir_extended + "/ " + str(item) + ".png")
            print(len(imagelist))
            np.savez_compressed(base_dir + "/op" + str(opacity), imagelist)


def distance(original: np.ndarray, modified: np.ndarray):

    #loop over all pixels in the new and original image, calculate the absolute difference and normalize the total.
    total = 0
    counter = 0
    for i in range(len(original)):
        if i % 10 == 0:
            counter += 1
            for j in range(len(original[i])):
                for k in range(len(original[i][j])):
                    #print("ORIGINAL: " + str(original[i][j][k]) + ", MODIFIED: " + str(modified[i][j][k]))
                    total += abs(original[i][j][k] - modified[i][j][k])

    factor = (counter * len(original[i]) * len(original[i][j]))
    print("factor : ", factor)
    return total / factor


def show_result(image):

    cv2.imshow("Digit", (image[0] * 255).astype("uint8"))
    cv2.waitKey(0)

def createnumbernoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_noise("numbers", "number_noise", x_test, 10, True, 0)

def createbarnoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_noise("bars", "bar_noise", x_test, 1, False, 0)

def createrandompixelnoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_random_pixels(28, x_test, True, 0)

def removepixels():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    remove_pixels(x_test, True)


def createdifferentrotations(noise, output_dir, length):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for x in range(0, 180):
        if x % 30 == 0:
            insert_noise(noise, output_dir + "/rotation_" + str(x), x_test, length, True, x)

def createrandomshapenoise():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    insert_shapes(x_test, True)

#createrandomshapenoise()