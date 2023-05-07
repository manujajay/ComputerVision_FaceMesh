# ComputerVision_FaceMesh
Detecting 400+ landmarks on faces using Computer Vision (OpenCV) and Mediapipe. A part of a wider project. 

## Install Dependencies
Pip install opencv-python and mediapipeline

## Run the script and set preferences

1. set:
    cap = cv2.VideoCapture(0) 
    to either (0) using your own webcame
    Or specify a path in the brackets ("path/to/video") to use a video file

2. Comment out:
    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    0.7, (0, 255, 0), 1)
    If you dont want see precise pixel locations of each point and just want 
    to see dot on a face mesh

Credit to [freeCodeCamp.org](https://www.youtube.com/watch?v=01sAkU_NvOY) and 
[Murtaza Hassan](https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A)

A few pieces are now deprecated from their original code so this repo contains the up-to date working modifications included too.