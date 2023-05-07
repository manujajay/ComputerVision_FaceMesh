import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, staticImage = False, maxFaces = 2, redefineLms = False, minDetectionCon = 0.5, minTrackingCon = 0.5 ):

        self.staticImage = staticImage
        self.maxFaces = maxFaces
        self.redefineLms = redefineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon
        
        self.mpDraw = mp.solutions.drawing_utils # This will help us draw on the faces
        # we could do this manually, but they create connetions between the points too 
        # So definitely a useful function to use (By Google I think?)
        self.drawSpecs = self.mpDraw.DrawingSpec(color = (0, 255, 0), thickness = 1, circle_radius = 1) # This is the spec for the drawing
        self.mpFaceMesh = mp.solutions.face_mesh # This is the face mesh model
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticImage,
                                                self.maxFaces,
                                                self.redefineLms,
                                                self.minDetectionCon,
                                                self.minTrackingCon) # We can detect multiple faces
 

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # We need to convert to RGB (faceMesh only takes RGB)
        
        self.results = self.faceMesh.process(self.imgRGB) # This will process the image and give us the results
        faces = [] # List of all the faces
        if self.results.multi_face_landmarks: # If we have detected faces
            for faceLms in self.results.multi_face_landmarks: # For each face
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms,
                                                self.mpFaceMesh.FACEMESH_TESSELATION,
                                                self.drawSpecs, self.drawSpecs) # Draw the connections on the face
                    face = [] # To store landmark for a single face
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
                        
                        # We can print actual pixel coordinates on the image so we can see where the landmarks are
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    0.7, (0, 255, 0), 1)
                        # print(id, x,y) # Print the pixel coordinates amd the id of the landmark
                        # The id shows that there are 468 landmarks
                        # We can use these landmarks to detect things like the eyes, nose, mouth, etc.
                        face.append([x,y])
                    faces.append(face)
        return img, faces  


def main () :
    pTime = 0
    cap = cv2.VideoCapture(0) # 0 is the webcam
    # Specify a path to a video to use a video (../Videos/3.mp4)

    detector = FaceMeshDetector()

    while True :
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (700, 420))
        img, faces = detector.findFaceMesh(img)

        # if len(faces) != 0 :
        #     print(len(faces))

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'fps : {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

if __name__ == "__main__" :
    main()