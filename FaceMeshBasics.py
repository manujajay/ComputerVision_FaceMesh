import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/3.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils # This will help us draw on the faces
# we could do this manually, but they create connetions between the points too 
# So definitely a useful function to use (By Google I think?)
mpFaceMesh = mp.solutions.face_mesh # This is the face mesh model
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # We can detect multiple faces
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2) # This is the spec for the drawing
# On HD videos, the thickness of 1 is too small, so we increase it to 2 (so we can see it better)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # We need to convert to RGB (faceMesh only takes RGB)
    results = faceMesh.process(imgRGB) # This will process the image and give us the results
    if results.multi_face_landmarks: # If we have detected faces
        for faceLms in results.multi_face_landmarks: # For each face
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec) # Draw the connections on the face
            for id, lm in enumerate(faceLms.landmark): # For each landmark
                # print(lm) # Print the landmark - this will give us the x,y,z coordinates
                # We can use these coordinates to do things like detect if the eyes are open
                # or closed, or if the mouth is open or closed, etc.
                # We can also use these coordinates to detect if the person is smiling or not
                # Right now they are normalized from 0 to 1, so we need to convert them to pixels
                ih, iw, ic = img.shape # Get the height, width, and channels of the image
                x, y = int(lm.x*iw), int(lm.y*ih) # Convert the normalized coordinates to pixels
                # We are multiplying by the width and height because the normalized coordinates
                # are from 0 to 1, so we need to convert them to pixels
                print(id, x,y) # Print the pixel coordinates amd the id of the landmark
                # The id shows that there are 468 landmarks
                # We can use these landmarks to detect things like the eyes, nose, mouth, etc.
                    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    