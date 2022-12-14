import cv2
import os

facedata = os.path.abspath("test/haarcascade_frontalface_alt.xml")
cascade = cv2.CascadeClassifier(facedata)

WIDTH = 256
HEIGHT = 256

# Inspirasi kode: https://www.geeksforgeeks.org/cropping-faces-from-images-using-opencv-python/
# https://stackoverflow.com/questions/48512532/cropping-faces-from-an-image-using-opencv-in-python


def facecropFolder(folder, dest):
    global facedata, cascade

    for fileName in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fileName))

        minisize = (img.shape[1], img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        if len(faces) != 0:
            f = faces[0]
            x, y, w, h = [v for v in f]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255))
            sub_face = img[y+2:y+h, x+2:x+w]

            file_name = "test/CroppedDataset/cropped_{}".format(fileName)
            cv2.imwrite(file_name, sub_face)


def facecropImage(img):
    global facedata, cascade

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)

    if len(faces) != 0:

        print("Face detected!")
        f = faces[0]

        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255))
        sub_face = img[y+2:y+h, x+2:x+w]

        file_name = "test/cropped_uji.jpg"
        cv2.imwrite(file_name, sub_face)

        return cv2.resize(
            sub_face, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
        )

    else:
        print("Face not detected")
        return img


if __name__ == "__main__":
    # Crop dataset, place into /test/cropped
    
    folder = input("Masukkan naam folder : ")
    parent = input("Masukkan folder parent ('Dataset' atau 'Datatest') : ")
    dataset_path = "test/parent/" + folder

    facecropFolder(os.path.abspath(dataset_path))

