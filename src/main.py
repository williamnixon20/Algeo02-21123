import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

img_width = 254
img_height = 254

path = "test/dataset/"
dataset_images_matrix = []

for photo_dir in os.listdir(path):
    image_converted = (
    Image.open(os.path.join(path, photo_dir))
    .resize((img_height, img_width), Image.ANTIALIAS)
    .convert('L'))
    image_matrix = np.array(image_converted).flatten()
    dataset_images_matrix.append(image_matrix)

banyak_dataset = len(dataset_images_matrix)
running_sum_face = np.zeros((1, img_height * img_width))

for photo in dataset_images_matrix:
    running_sum_face += photo

mean_face = (running_sum_face / len(dataset_images_matrix)).flatten()
plt.imshow(mean_face.reshape(img_height, img_width), cmap='gray')
plt.title("Average face")
plt.show()

face_normalized_matrix = np.ndarray(shape=(banyak_dataset, img_height*img_width))

for idx in range(banyak_dataset):
    face_normalized_matrix[idx] = np.subtract(dataset_images_matrix[idx], mean_face)
    plt.imshow(dataset_images_matrix[idx].reshape(img_height, img_width), cmap='gray')
    plt.title('Before norm')
    plt.show()
    plt.imshow(face_normalized_matrix[idx].reshape(img_height, img_width), cmap='gray')
    plt.title("After norm")
    plt.show()