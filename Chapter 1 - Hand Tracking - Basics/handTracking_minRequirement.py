"""
    Name: Rahmani Dibansa
    Date: 21st of August 2022
    Description:
        This python program is made by following through a tutorial. I have
        added in extensive explanations of how I understood the tutorial.
    Reference(s):
        FreeCodeCamp. (Hassan, M. , 2021). Advanced Computer Vision with Python - Full Course.
        Retrieved from: https://www.youtube.com/watch?v=01sAkU_NvOY&t=257s
"""

# Libraries needed:
# pip install cv2
# pip install mediapipe
import cv2
import mediapipe as mp
import time


# Open the webcam and capture it using opencv
captureWebcam = cv2.VideoCapture(0)

# Initialize the mediapipe hands
# This is for detecting hands
mpHands = mp.solutions.hands

# Initialize the drawing utility of mediapipe
# This will help in drawing the landmark and its connections
mpDraw = mp.solutions.drawing_utils

# The mediapipe hands has these parameters for detection:
# static_image_mode (default is False):
#               - If set to "True", it will keep on detecting the whole time
#                 and will not track. This will take more time
#               - If set to "False", it will detect and track depending on the
#                 confidence rates set on the parameters.
#
# max_num_hands (default is 2): for the max number of hands that should be detected
#
# min_detection_confidence (default is 0.5): the confidence rate for the detection
#
# min_tracking_confidence (default is 0.5): the confidence rate for tracking. If it goes below
#                                           the set confidence, it will opt to detect instead of
#                                           track.
handsDetection = mpHands.Hands()


# Time references to calculate Frame per Second.
currentTime = 0
previousTime = 0

# While the webcam is being captured, execute the block of code inside this
while captureWebcam.isOpened():
    # Read the video from the webcam
    # If successful, take its frame and set success into True
    # If not, set success intro False ( But I don't know what the value of videoFrame will be )
    success, videoFrame = captureWebcam.read()

    # To understand it more, try these:    
    #print( " Success: " , success )
    #print( " Video Frame : " , videoFrame )

    # Since the mediapipe process only accepts RGB images, we would need to convert the
    # video frame from BGR -> RGB.
    imgRGB = cv2.cvtColor( videoFrame, cv2.COLOR_BGR2RGB)
    # Process the RGB image to detect for the presence of hand(s)
    processResult = handsDetection.process( imgRGB)

    # To see what is inside the result, try printing
    # If there are no hand, it will print None
    # If there are, then it will print the coordinates of the hand(s)
    # print( processResult.multi_hand_landmarks)


    # If there are hands detected, execute the code inside this condition
    if processResult.multi_hand_landmarks:
        # Take the handLandmarks from the process result.
        # This will iterate through every detected hands.
        for handLandmarks in processResult.multi_hand_landmarks:
            """
            # Draw dots/circles on the landmarks
            # mpDraw.draw_landmarks( videoFrame, handLandmarks)

            # Draw the dots on the landmarks and connect it using mpHands.HAND_CONNECTIONS
            mpDraw.draw_landmarks( videoFrame, handLandmarks, mpHands.HAND_CONNECTIONS)
            """
            # Take each landmark from handLandmarks.
            # landmarkIndex: the ID of the specific landmark.
            # landmarkCoordinates: the x, y, and z coordinates of a specific landmark
            for landmarkIndex, landmarkCoordinates in enumerate( handLandmarks.landmark ):
                # Get the shape of the videoFrame
                # It returns a tuple of the number of rows, columns, and channels (if the image is color)
                # rows: number of elements in the outer layer
                # columns: number of elements inside the inner layer
                # channels: the number of colors present in the image. If it is grayscale, then it will be 0
                height, width, channels = videoFrame.shape
                centerX, centerY = int( landmarkCoordinates.x*width), int(landmarkCoordinates.y*height)
                
                # To see the hand landmarks along with their description, open the HandLandmarks-Reference.png
                # 0. WRIST
                # 1. THUMB_CMC  5. INDEX_FINGER_MCP     9. MIDDLE_FINGER_MCP    13. RING_FINGER_MCP     17. PINKY_FINGER_MCP
                # 2. THUMB_MCP  6. INDEX_FINGER_PIP     10. MIDDLE_FINGER_PIP   14. RING_FINGER_PIP     18. PINKY_FINGER_PIP
                # 3. THUMB_IP   7. INDEX_FINGER_DIP     11. MIDDLE_FINGER_DIP   15. RING_FINGER_DIP     19. PINK_FINGER_DIP
                # 4. THUMB_TIP  8. INDEX_FINGER_TIP     12. MIDDLE_FINGER_TIP   16. RING_FINGER_TIP     20. PINKY_FINGER_TIP
                if landmarkIndex == 4:
                    cv2.circle( videoFrame, (centerX, centerY), 25, (255, 0, 255), cv2.FILLED )

                # Draw the dots on the landmarks and connect it using mpHands.HAND_CONNECTIONS
                mpDraw.draw_landmarks( videoFrame, handLandmarks, mpHands.HAND_CONNECTIONS)
    # Calculate the FPS
    currentTime = time.time()
    fps = 1 / ( currentTime - previousTime )
    previousTime = currentTime


    # putText format: image, text, location/coordinates, font style, scale , color, thickness
    cv2.putText( videoFrame, str(int((fps))), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2 )
    
    # Use opencv to display the videoFrame
    cv2.imshow( "Advanced Computer Vision with Python: Chapter 1", videoFrame )

    exitESC = cv2.waitKey(30) & 0xff
    if exitESC == 27:
        break

    
# Terminate webcam capture
captureWebcam.release()
cv2.destroyAllWindows()
