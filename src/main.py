import time
import cv2
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
from util import *

folder_training_path = ""
image_test_path = ""
turn_on_cam = False

CAM_WIDTH = 350
CAM_HEIGHT = 350
frameSize = (CAM_WIDTH, CAM_HEIGHT)
video_capture = cv2.VideoCapture(0)
time.sleep(5)

imagesData = ""
meanFace = ""
eigenFaces = ""
databaseWeighted = ""
imagesNormal = ""


def SetupFile():
    global folder_training_path, image_test_path
    folder_layout = [
        [sg.T("Silahkan pilih folder training dataset (mengandung banyak citra uji)")],
        [
            sg.Text("Choose a folder: "),
            sg.Input(key="-IN2-", change_submits=True),
            sg.FolderBrowse(key="-IN-2"),
        ],
    ]
    file_layout = [
        [sg.T("Silahkan pilih file citra yang ingin anda uji")],
        [
            sg.Text("Choose a file: "),
            sg.Input(key="-IN1-", change_submits=True),
            sg.FileBrowse(key="-IN-1"),
        ],
    ]

    button_layout = [
        [sg.T("Hmm, males ah ribet. Pakai konfigurasi built in aja deh!")],
        sg.Button("Default"),
    ]

    layout = [folder_layout, file_layout, [sg.Button("Submit")], button_layout]

    window = sg.Window(
        "Tubes Algeo - 2, EIGENFACE!",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
    )

    while True:
        event, values = window.read(timeout=10)

        if event == sg.WIN_CLOSED:
            exit()
        if event == "Submit":
            print("I received this as image dir" + values["-IN-1"])
            image_test_path = values["-IN-1"]

            print("I received this as training folder dir" + values["-IN-2"])
            folder_training_path = values["-IN-2"]
            break
        if event == "Default":
            image_test_path = os.path.abspath("test/gambar.jpg")
            folder_training_path = os.path.abspath("test/dataset")
            break
    window.close()


def PromptTurnOnCam():
    global turn_on_cam
    button_layout = [
        [
            sg.T(
                "Apakah anda mau menyalakan cam agar foto anda diambil setiap 30 detik?"
            )
        ],
        [sg.Button("Ya")],
        [sg.Button("Tidak")],
    ]

    layout = [button_layout]

    window = sg.Window(
        "Tubes Algeo - 2, EIGENFACE!",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
    )

    while True:
        event, values = window.read(timeout=10)

        if event == sg.WIN_CLOSED:
            exit()
        if event == "Ya":
            turn_on_cam = True
            break
        if event == "Tidak":
            turn_on_cam = False
            break
    window.close()


def DisplayResult():
    global image_test_path
    reference_pic = [
        [sg.Text("Foto Uji", size=(60, 1), justification="center")],
        [sg.Image(key="col1")],
    ]
    colgalery1 = sg.Column(reference_pic, element_justification="center")

    result_pic = [
        [sg.Text("Kamu mirip dia lho...", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="col2")],
    ]
    colgalery2 = sg.Column(result_pic, element_justification="center")

    file_layout = [
        [sg.T("Ingin mengganti citra uji?")],
        [
            sg.Text("Choose a file: "),
            sg.Input(key="-IN1-", change_submits=True),
            sg.FileBrowse(key="-IN-1"),
        ],
        [sg.Button("Submit")],
    ]

    colslayout = [[colgalery1, colgalery2], file_layout]
    window = sg.Window(
        "Tubes Algeo - 2, EIGENFACE!",
        colslayout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
    )
    need_refresh = True
    pic_displayed = True
    while True:
        event, values = window.read(timeout=10)
        if not pic_displayed:
            img = getSimilarPicture(image_test_path)
            print(image_test_path)
            window["col2"].update(data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True
        if need_refresh:
            need_refresh = False
            pic_displayed = False
            print(image_test_path)
            image = Image.open(image_test_path)
            window["col1"].update(data=ImageTk.PhotoImage(image))
        if event == sg.WIN_CLOSED:
            window.close()
            exit()
        if event == "Submit":
            pic_displayed = False
            print("I received this as image dir" + values["-IN-1"])
            image_test_path = values["-IN-1"]
            need_refresh = True


def DisplayResultCam():
    image_test_path = os.path.abspath("test/gambar.jpg")
    global video_capture
    camera_frame = [
        [
            sg.Text(
                "Cheese! Fotomu kita sample tiap 30 detik :D",
                size=(60, 1),
                justification="center",
            )
        ],
        [sg.Image(filename="", key="col1")],
    ]
    colgalery1 = sg.Column(camera_frame, element_justification="center")

    picture_frame = [
        [sg.Text("Cakep bener ih", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="col2")],
    ]
    colgalery2 = sg.Column(picture_frame, element_justification="center")

    similar_frame = [
        [sg.Text("Kamu mirip sama dia..", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="col3")],
    ]
    colgalery3 = sg.Column(similar_frame, element_justification="center")
    layout = [[colgalery1, colgalery2, colgalery3]]

    window = sg.Window(
        "Tubes Algeo - 2, EIGENFACE!",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
    )
    start_time = time.time()
    photo_taken = None
    first_loop = True
    pic_displayed = True
    while True:
        event, values = window.read(timeout=10)

        # get camera frame
        ret, frameOrig = video_capture.read()
        frame = cv2.resize(frameOrig, frameSize)

        if event == sg.WIN_CLOSED:
            video_capture.release()
            cv2.destroyAllWindows()
            break

        # fpsInfo = "TIME: " + str((time.time() - start_time))
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 255, 255), 1)
        if not pic_displayed:
            img = getSimilarPicture(image_test_path)
            window["col3"].update(data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True
        if (time.time() - start_time) > 30 or first_loop:
            first_loop = False
            start_time = time.time()
            print("saving")
            photo_taken = frame
            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(imageRGB)
            im.save("test/gambar.jpg")
            pic_displayed = False

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["col1"].update(data=imgbytes)

        imgbytes = cv2.imencode(".png", photo_taken)[1].tobytes()
        window["col2"].update(data=imgbytes)


def Loading():
    global imagesData, imagesNormal, meanFace, eigenFaces, databaseWeighted
    imagesNormal = GetImagesNorm(folder_training_path)
    imagesData = GetImagesTrain(folder_training_path)
    meanFace = GetMeanFace(imagesData)
    normalizedData = GetNormalized(imagesData, meanFace)
    covMatrix = GetCovariance(normalizedData)

    (
        eigenvalues,
        eigenvectors,
    ) = GetEigenInfo(covMatrix)
    eigenvalues, eigenvectors = sortEigen(eigenvalues, eigenvectors)
    eigenFaces = GetEigenFaces(eigenvectors, normalizedData)
    databaseWeighted = getWeighted(eigenFaces, normalizedData)


def getSimilarPicture(absPath):
    normalizedTestImg = getNormalizedTestImage(absPath, meanFace)
    testWeighted = getWeighted(eigenFaces, normalizedTestImg)
    image_index, value = getEuclideanDistance(databaseWeighted, testWeighted)
    img = imagesNormal[image_index]
    return cv2.resize(img, (CAM_HEIGHT, CAM_WIDTH), interpolation=cv2.INTER_AREA)


SetupFile()
Loading()
PromptTurnOnCam()
if turn_on_cam:
    DisplayResultCam()
else:
    DisplayResult()
