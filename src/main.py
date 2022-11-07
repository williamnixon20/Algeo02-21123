import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

img_width = 254
img_height = 254

path = "test/dataset/"
dataset_images = []

for photo_dir in os.listdir(path):
    dataset_images.append(
    Image.open(os.path.join(path, photo_dir))
    .resize((img_height, img_width), Image.ANTIALIAS)
    .convert('L'))

counter = 0
for photo in dataset_images:
    photo.show()
    counter += 1
    if (counter == 5):
        break