# Vehicle-Counting-and-Classification

**Features**
Object detection using Single Shot MultiBox Detector (SSD) with MobileNet as the base architecture.\n
Color detection using the K-Nearest Neighbors (KNN) algorithm.\n
Lightweight and efficient models suitable for real-time applications.\n
Customizable and extensible for further experimentation.\n

**Requirements**
The following python libraries are requried to run the project:
    numpy
    scipy
    scikit-image
    tensorflow
    opencv-python
    packaging

**Running the Project**
You can test the project by one of these commands. Program takes an input argument 'imshow' or 'imwrite':

      python3 vehicle_detection_main.py imshow
      python3 vehicle_detection_main.py imwrite
- *imshow*  : shows the processed frames as an video on screen.
- *imwrite* : saves the processed frames as an output video in the project root folder.
