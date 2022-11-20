import time
import cv2
import PySimpleGUI as sg
import os
import threading
from PIL import Image, ImageTk
from util import *

image_test_path = os.path.abspath("test/gambar.jpg")
folder_training_path = os.path.abspath("test/dataset")
turn_on_cam = False
start_time = 0
total_time = 0
INTELLI_CROP = True

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
hasLoaded = False
# STYLING
sg.theme('LightGrey1')
titleFont = ('Quicksand', 16)
generalFont = ("Quicksand", 12)
labelFont = ("Quicksand", 11)
inputFont = ("Quicksand", 9)


def SetupFile():
    global folder_training_path, image_test_path

    header_layout = [
        sg.Column(
            [
                [sg.Text("Face Recognition", font=titleFont)],
            ],

            element_justification="left",
            vertical_alignment="center",
        ),

        sg.Column(
            [
                [sg.Button("Home", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))],

            ],
            element_justification="right",
            vertical_alignment="center",
            expand_x=True,

        )
    ]
    folder_layout = [
        [sg.T("Choose Dataset Folder")],
        [
            sg.Text("Dataset Folder : ", font=labelFont),
            sg.Input(key="-IN2-", change_submits=True, border_width=0.1,
                     font=inputFont, text_color='#f1356d', size=(30, 1)),
            sg.FolderBrowse(key="-IN-2", button_text='Choose Folder',
                            size=(15, 1), font=inputFont),
        ],
    ]
    file_layout = [
        [sg.T("Choose Test Image")],
        [
            sg.Text("Test Image : ", font=labelFont),
            sg.Input(key="-IN1-", change_submits=True, border_width=0.1,
                     text_color='#f1356d', font=inputFont, size=(30, 1)),
            sg.FileBrowse(key="-IN-1", button_text='Choose File',
                          size=(15, 1), font=inputFont),
        ],
    ]

    submit_button_layout = [
        sg.Column(
            [
                [sg.Button("Submit", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))]
            ],
            element_justification="center",
            expand_x=True
        )
    ]

    default_button_layout = [
        sg.Column(
            [
                [sg.Button("Use Default Test Path", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))]
            ],
            element_justification="center",
            expand_x=True,
        )
    ]
    layout = [header_layout, folder_layout, file_layout,
              submit_button_layout, default_button_layout]

    window = sg.Window(
        "CapeFace/Home",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=True,
        return_keyboard_events=True,
        location=(100, 100),
        font=generalFont,
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        use_default_focus=False
    )

    while True:
        event, values = window.read(timeout=10)

        if event == sg.WIN_CLOSED:
            exit()
        if event == "Submit":
            if (len(values["-IN-1"]) != 0):
                image_test_path = values["-IN-1"]
            if (len(values["-IN-2"]) != 0):
                folder_training_path = values["-IN-2"]
            break
        if event == "Use Default Test Path":
            image_test_path = os.path.abspath("test/gambar.jpg")
            folder_training_path = os.path.abspath("test/dataset")
            break
    window.close()


def PromptTurnOnCam():
    global turn_on_cam

    header_layout = [
        sg.Column(
            [
                [sg.Text("Face Recognition", font=titleFont)],
            ],

            element_justification="left",
            vertical_alignment="center",
        ),

        sg.Column(
            [
                [sg.Button("Home", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))],

            ],
            element_justification="right",
            vertical_alignment="center",
            expand_x=True,

        )
    ]

    text_layout = [
        [
            sg.T(
                "Turn on camera? (Photo will be taken every 30 seconds)",
                font=labelFont
            ),
        ],
    ]

    on_camera_layout = [
        sg.Column(
            [
                [sg.Button("Ya", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))]
            ],
            element_justification="center",
            expand_x=True
        )
    ]

    off_camera_layout = [
        sg.Column(
            [
                [sg.Button("Tidak", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))]
            ],
            element_justification="center",
            expand_x=True,
        )
    ]

    layout = [header_layout, text_layout, on_camera_layout, off_camera_layout]

    window = sg.Window(
        "CapeFace/Prompt",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
    )

    while True:
        event, values = window.read(timeout=100)

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
    header_layout = [
        sg.Column(
            [
                [sg.Text("Face Recognition", font=titleFont)],
            ],

            element_justification="left",
            vertical_alignment="center",
        ),
        sg.Column(
        [
            [sg.Text(f"Time needed to process dataset: {total_time}")]
        ]),

        sg.Column(
            [
                [sg.Button("Home", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))],

            ],
            element_justification="right",
            vertical_alignment="center",
            expand_x=True,

        )
    ]
    reference_pic = [
        [sg.Text("Test Image", size=(60, 1), justification="center")],
        [sg.Image(key="col1")],
    ]
    colgalery1 = sg.Column(reference_pic, element_justification="center")

    result_pic = [
        [sg.Text("Closest Result", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="col2")],
    ]
    colgalery2 = sg.Column(result_pic, element_justification="center")

    file_layout = [
        [sg.T("Choose Another Test Image")],
        [
            sg.Text("Choose File : "),
            sg.Input(key="-IN1-", change_submits=True),
            sg.FileBrowse(key="-IN-1"),
        ],
        [sg.Button("Submit")],
    ]

    colslayout = [header_layout, [colgalery1, colgalery2], file_layout]
    window = sg.Window(
        "CapeFace/Result",
        colslayout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
    )
    need_refresh = True
    pic_displayed = True
    while True:
        event, values = window.read(timeout=100)
        if not pic_displayed:
            img = getSimilarPicture(image_test_path)
            window["col2"].update(
                data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True
        if need_refresh:
            need_refresh = False
            pic_displayed = False
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
    global video_capture, total_time

    header_layout = [
        sg.Column(
            [
                [sg.Text("Face Recognition", font=titleFont)],
            ],

            element_justification="left",
            vertical_alignment="center",
        ),
        sg.Column(
        [
            [sg.Text(f"Time needed to process dataset: {total_time}")]
        ]
        ),
        sg.Column(
            [
                [sg.Button("Home", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))],

            ],
            element_justification="right",
            vertical_alignment="center",
            expand_x=True,

        )
    ]

    camera_frame = [
        [
            sg.Text(
                "Photo will be taken every 20 seconds",
                justification="center",
            ),
            sg.Text('', key='_time_', size=(20, 1))
        ],
        [sg.Image(filename="", key="col1")],
    ]
    colgalery1 = sg.Column(camera_frame, element_justification="center")

    picture_frame = [
        [sg.Text("Photo from Camera", justification="center")],
        [sg.Image(filename="", key="col2")],
    ]
    colgalery2 = sg.Column(picture_frame, element_justification="center")

    similar_frame = [
        [sg.Text("Closest Result", justification="center")],
        [sg.Image(filename="", key="col3")],
    ]
    colgalery3 = sg.Column(similar_frame, element_justification="center")
    layout = [header_layout, [colgalery1, colgalery2, colgalery3]]

    window = sg.Window(
        "CapeFace/Camera",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
    )
    start_time = time.time()
    photo_taken = None
    first_loop = True
    pic_displayed = True
    while True:
        event, values = window.read(timeout=150)

        # get camera frame
        ret, frameOrig = video_capture.read()
        frame = cv2.resize(frameOrig, frameSize)

        if event == sg.WIN_CLOSED:
            video_capture.release()
            cv2.destroyAllWindows()
            break

        if not pic_displayed:
            img = getSimilarPicture(image_test_path)
            window["col3"].update(
                data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True

        if (time.time() - start_time) > 20 or first_loop:
            first_loop = False
            start_time = time.time()
            photo_taken = frame
            imageRGB = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(imageRGB)
            im.save("test/gambar.jpg")
            pic_displayed = False

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["col1"].update(data=imgbytes)

        imgbytes = cv2.imencode(".png", photo_taken)[1].tobytes()
        window["col2"].update(data=imgbytes)

        timeInfo = "TIME: " + str(round((time.time() - start_time), 1))
        window["_time_"].update(timeInfo)


def Loading():
    global imagesData, imagesNormal, meanFace, eigenFaces, databaseWeighted, hasLoaded
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
    hasLoaded = True
    return


def getSimilarPicture(absPath):
    normalizedTestImg = getNormalizedTestImage(absPath, meanFace, INTELLI_CROP)
    testWeighted = getWeighted(eigenFaces, normalizedTestImg)
    image_index, value = getEuclideanDistance(databaseWeighted, testWeighted)
    img = imagesNormal[image_index]
    return cv2.resize(img, (CAM_HEIGHT, CAM_WIDTH), interpolation=cv2.INTER_AREA)


def LoadingScreen():
    global hasLoaded, start_time, total_time
    loading_col = [sg.Column(
        [
            [sg.Text("Please wait while we are loading..", font=titleFont)],
            [sg.Text('', key='_time_', size=(20, 1))]
        ]
    )]

    window = sg.Window(
        "CapeFace/Camera",
        [loading_col],
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        location=(100, 100),
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
    )
    while True:
        event, values = window.read(timeout=150)

        if event == sg.WIN_CLOSED:
            video_capture.release()
            cv2.destroyAllWindows()
            break

        if hasLoaded == True:
            total_time = round((time.time()-start_time), 2)
            break
        timeInfo = "TIME: " + str(round((time.time() - start_time), 2))
        window["_time_"].update(timeInfo)
    window.close()


SetupFile()
PromptTurnOnCam()
start_time = time.time()
threading.Thread(target=Loading,
                 args=(),
                 daemon=True).start()
LoadingScreen()
if turn_on_cam:
    DisplayResultCam()
else:
    DisplayResult()
