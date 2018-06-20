import copy
import cv2
import os
from PIL import Image
import math
import random
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

overlay_path = os.path.join(os.path.dirname(__file__), '../overlays/' + 'shapes' + '/' + str(1))
overlay = Image.open(overlay_path)
pixeldata = list(overlay.getdata())
for angle in range(0, 360):
    overlay_copy = Image.new(overlay.mode, overlay.size)
    overlay_copy.putdata(pixeldata)
    overlay_copy = overlay_copy.rotate(angle, expand=True)
    overlay_copy.save("../test/" + str(angle) + ".png")