import time
import cv2
import PySimpleGUI as sg
import os
import threading
from PIL import Image, ImageTk
from util import *

image_test_path = ""
folder_training_path = ""
turn_on_cam = False
start_time = 0
total_time = 0
INTELLI_CROP = True
THRESHOLD_PERSON = 80000000
THRESHOLD_DATASET = 130000000

CAM_WIDTH = 350
CAM_HEIGHT = 350
frameSize = (CAM_WIDTH, CAM_HEIGHT)
video_capture = cv2.VideoCapture(0)
time.sleep(5)
cameraTime = 15

imagesData = ""
meanFace = ""
eigenFaces = ""
databaseWeighted = ""
imagesNormal = ""

eigenvectors = ""
covMatrix = "" 
normalizedData = ""

hasLoaded = False

goToHome = False
err = False
empty_test_err = False

# STYLING
sg.theme('LightGrey1')
titleFont = ('Quicksand', 16)
generalFont = ("Quicksand", 12)
labelFont = ("Quicksand", 11)
inputFont = ("Quicksand", 9)

def move_center(window):
    screen_width, screen_height = window.get_screen_dimensions()
    win_width, win_height = window.size
    x, y = (screen_width - win_width)//2, (screen_height - win_height)//2
    window.move(x, y)

def SetupFile():
    global folder_training_path, image_test_path, goToHome, err, empty_test_err

    errWarning = ""
    testWarning = ""

    if (err):
        errWarning = " (Required or Choose Default Path)"
    
    if(empty_test_err):
        testWarning = " (Required or Choose Default Path)"

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
        [sg.T("Choose Dataset Folder" + errWarning)],
        [
            sg.Text("Dataset Folder : ", font=labelFont),
            sg.Input(key="-IN2-", change_submits=True, border_width=0.1,
                     font=inputFont, text_color='#f1356d', size=(50, 1), readonly=True),
            sg.FolderBrowse(key="-IN-2", button_text='Choose Folder',
                            size=(15, 1), font=inputFont),
        ],
    ]
    file_layout = [
        [sg.T("Choose Test Image" + testWarning)],
        [
            sg.Text("Test Image : ", font=labelFont),
            sg.Input(key="-IN1-", change_submits=True, border_width=0.1,
                     text_color='#f1356d', font=inputFont, size=(50, 1), readonly=True),
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
        font=generalFont,
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        use_default_focus=False,
        finalize=True
    )

    move_center(window)

    while True:
        event, values = window.read(timeout=10)

        if event == sg.WIN_CLOSED:
            exit()
        if event == "Submit":

            if (len(values["-IN-1"]) != 0):
                image_test_path = values["-IN-1"]

            if (len(values["-IN-2"]) != 0):
                folder_training_path = values["-IN-2"]
                err = False

            else:
                err = True

            break
        if event == "Use Default Test Path":
            err = False

            if (len(values["-IN-1"]) != 0):
                image_test_path = values["-IN-1"]

            else:  
                image_test_path = os.path.abspath("test/gambar.jpg")

            folder_training_path = os.path.abspath("test/dataset")
            break

    window.close()


def PromptTurnOnCam():
    global turn_on_cam, goToHome, cameraTime, empty_test_err

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
                "Turn on camera? (Photo will be taken every " + str(cameraTime) + " seconds)",
                font=labelFont
            ),
        ],
    ]

    on_camera_layout = [
        sg.Column(
            [
                [sg.Button("Yes", border_width=0, mouseover_colors=(
                    '#000000', '#FFFFFF'), highlight_colors=('#000000', '#FFFFFF'))]
            ],
            element_justification="center",
            expand_x=True
        )
    ]

    off_camera_layout = [
        sg.Column(
            [
                [sg.Button("No", border_width=0, mouseover_colors=(
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
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
        finalize=True
    )

    move_center(window)

    while True:
        event, values = window.read(timeout=100)

        if event == sg.WIN_CLOSED:
            exit()
        if event == "Yes":
            empty_test_err = False
            turn_on_cam = True
            break
        if event == "No":
            if (len(image_test_path) == 0):
                empty_test_err = True
            
            else:
                empty_test_err = False

            turn_on_cam = False
            break
        
        if event == "Home":
            empty_test_err = False
            goToHome = True
            break

    window.close()


def DisplayResult():
    global image_test_path, goToHome
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

    label_layout = [
        [sg.Text(f"Time needed to process dataset: {total_time}")],
        [sg.Text('Euclidean distance: ', key='_dist_', size=(50, 1))],
        [sg.Text('Info: ', key='_info_', size=(50, 1))]
       
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
            sg.Input(key="-IN1-", change_submits=True, font=inputFont, text_color='#f1356d', readonly=True),
            sg.FileBrowse(key="-IN-1"),
        ],
        [sg.Button("Submit")],
    ]

    colslayout = [header_layout, label_layout, [colgalery1, colgalery2], file_layout]
    window = sg.Window(
        "CapeFace/Result",
        colslayout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
        finalize=True,
        location=(100,100)
    )

    need_refresh = True
    pic_displayed = True

    while True:
        event, values = window.read(timeout=100)
        if not pic_displayed:
            img, val = getSimilarPicture(image_test_path)
            window["col2"].update(
                data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True
            eucDist = "EUCLIDEAN DISTANCE: {}".format(str(round(val)))
            window["_dist_"].update(eucDist)
            info = "Person in database!"
            if (val > THRESHOLD_DATASET):
                info = "Person not in database"
            if (val > THRESHOLD_PERSON):
                info = "Picture is unrecognized, maybe not a person/in DB"
            window["_info_"].update(info)
        if need_refresh:
            need_refresh = False
            pic_displayed = False
            image = Image.open(image_test_path)
            image = image.resize(frameSize, Image.ANTIALIAS)
            window["col1"].update(data=ImageTk.PhotoImage(image))
        if event == sg.WIN_CLOSED:
            window.close()
            exit()
        if event == "Submit":
            pic_displayed = False
            print("I received this as image dir" + values["-IN-1"])
            image_test_path = values["-IN-1"]
            need_refresh = True
        
        if event == "Home":
            goToHome = True
            window.close()
            break


def DisplayResultCam():
    image_test_path = os.path.abspath("test/gambar.jpg")
    global video_capture, total_time, goToHome, cameraTime

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

    label_layout = [

                [sg.Text(f"Time needed to process dataset: {total_time}")],
                [sg.Text('Euclidean distance: ', key='_dist_', size=(50, 1))],
                [sg.Text('Info: ', key='_info_', size=(50, 1))]
    ]

    camera_frame = [
        [
            sg.Text(
                "Photo will be taken every " + str(cameraTime) + " seconds",
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
    layout = [header_layout, label_layout, [colgalery1, colgalery2, colgalery3]]

    window = sg.Window(
        "CapeFace/Camera",
        layout,
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
        location=(100,100)
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
            img, val = getSimilarPicture(image_test_path)
            window["col3"].update(
                data=ImageTk.PhotoImage(image=Image.fromarray(img)))
            pic_displayed = True
            
            eucDist = "EUCLIDEAN DISTANCE: {}".format(str(round(val)))
            window["_dist_"].update(eucDist)
            info = "Person in database!"
            if (val > THRESHOLD_DATASET):
                info = "Person not in database"
            if (val > THRESHOLD_PERSON):
                info = "Picture unrecognized, maybe not a person / not in database"
            window["_info_"].update(info)

        if (time.time() - start_time) > cameraTime or first_loop:
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

        if event == "Home":
            goToHome = True
            window.close()
            break


def Loading():
    global imagesData, imagesNormal, meanFace, eigenFaces, databaseWeighted, hasLoaded, eigenvectors, covMatrix, normalizedData
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

    # VERSI 1 ===========
    # eigenFaces = GetEigenFaces(eigenvectors, normalizedData)
    # databaseWeighted = getWeighted(eigenFaces, normalizedData)
    # ==================
    
    # versi 2 ============
    databaseWeighted = getEigenFaces2(eigenvectors, covMatrix)
    # =============

    hasLoaded = True
    return


def getSimilarPicture(absPath):
    normalizedTestImg = ""
    if (absPath.find("cropped") != -1):
        normalizedTestImg = getNormalizedTestImage(absPath, meanFace, False)
    else:
        normalizedTestImg = getNormalizedTestImage(absPath, meanFace, INTELLI_CROP)

    # VERSI 1
    # testWeighted = getWeighted(eigenFaces, normalizedTestImg)

    # ================

    # VERSI 2 ======
    testWeighted = getTestEigenFaces(eigenvectors, normalizedData, normalizedTestImg)

    # ===========
    image_index, value = getEuclideanDistance(databaseWeighted, testWeighted)
    img = imagesNormal[image_index]
    return cv2.resize(img, (CAM_HEIGHT, CAM_WIDTH), interpolation=cv2.INTER_AREA), value


def LoadingScreen():
    global hasLoaded, start_time, total_time

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
                [sg.Button("Home", border_width=0,
                           button_color=("#FFFFFF", '#FFFFFF'))],

            ],
            element_justification="right",
            vertical_alignment="center",
            expand_x=True,

        )
    ]

    loading_col = [sg.Column(
        [
            [sg.Text("Please wait while we are loading..", font=labelFont)],
            [sg.Text('', key='_time_', size=(50, 1))]
        ]
    )]

    window = sg.Window(
        "CapeFace/Camera",
        [header_layout, loading_col],
        no_titlebar=False,
        alpha_channel=1,
        grab_anywhere=False,
        return_keyboard_events=True,
        button_color=('#f1356d', '#FFFFFF'),
        titlebar_background_color='#f1356d',
        font=generalFont,
        finalize=True
    )

    move_center(window)

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

while True:

    image_test_path = ""
    folder_training_path = ""

    goToHome = False

    SetupFile()
    
    if(err):
        continue

    PromptTurnOnCam()

    if (empty_test_err):
        continue

    if (goToHome):
        continue

    start_time = time.time()
    threading.Thread(target=Loading,
                    args=(),
                    daemon=True).start()

    LoadingScreen()

    if turn_on_cam:
        DisplayResultCam()

    else:
        DisplayResult()
