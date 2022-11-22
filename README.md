# Tubes Algeo 2, Eigenfaces!

# Members
13521123	William Nixon 
13521148 	Johanes Lee 
13521170	Haziq Abiyyu Mahdy 

# How to run
1. Create a virtual environment folder and activate it. virtualenv venv, ./bin/scripts/Activate
2. Download requirements, pip install -r requirements.txt
3. Default will use files in test/cropped. You are done! Please do not use uncropped database and preprocess them first by cropping them. Follow step 4.
4. If you wanna use your own dataset, please crop them first!!! Your results will be much better if you follow this (we crop test images, if you dont crop the dataset, it won't perform well): 
    -  Delete everything on /test/cropped
    - Go to face_detector.py, change folder_path to your new dataset path. Run face_detector.py, ensure your dataset should now be in test/cropped and will now be cropped.
    - Run the GUI, select test/cropped as your new dataset and select your test image.
5. If you wanna use the face recognition feature, prepare your own dataset and do step (4).
6. Run main.py. Enjoy!

# Credits
We would like to thank the various resources that we have referenced to make this program possible. We also would like to thank the Yale and Kaggle for providing us with the training data we use. We do not claim to have made the auto-crop feature. It's all possible thanks to openCV's training data and geek4geeks example program. 
- https://www.geeksforgeeks.org/cropping-faces-from-images-using-opencv-python/
- https://stackoverflow.com/questions/48512532/cropping-faces-from-an-image-using-opencv-in-python

If you have difficulty 