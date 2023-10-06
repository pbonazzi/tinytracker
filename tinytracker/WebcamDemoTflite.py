import tensorflow as tf
import cv2
import numpy as np

import matplotlib.pyplot as plt
def draw_rectangle(image_shape, rect):
    # Create a black image
    image = np.zeros(image_shape, dtype=np.uint8)
    x, y, w, h = rect
    # Calculate the coordinates of the top-left and bottom-right points of the rectangle
    top_left = (x, y)
    bottom_right = (x + w, y + h)

    # Draw a white rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)

    return image

def crop_image(image, rect):
    # Unpack the rectangle coordinates
    x, y, w, h = rect

    # Crop the image based on the provided coordinates
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def show_face_detect(img, face):
    image = img.copy()
    x, y, w, h = face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image
def create_grid(g, w, h):
    x_start = g[0]*2/w - 1
    x_end = (g[0] + g[2])*2/w-1
    y_start = g[1]*2/h - 1
    y_end = (g[1] + g[3])*2/h-1
    linx = np.linspace(x_start,x_end,128)
    liny = np.linspace(y_start,y_end,128)
    return np.meshgrid(linx, liny)
def detect_eyes_and_face(image, static_crop = False):
    # Load the Haar cascade XML files for face and eye detection
    face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

    # Read the image using OpenCV


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if static_crop:
        largest_face = int(1920 / 2 - 96 * 2), int(1080 / 2 - 96 * 2), int(96 * 4), int(96 * 4)
    else:

        largest_face = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:1][0]

    x, y, w, h = largest_face



    # Extract the region of interest (ROI) for eyes within the face rectangle
    roi_gray = gray[y:y+h, x:x+w]


    # Detect eyes within the ROI

    debug_image = show_face_detect(image,largest_face)


    face = crop_image(image, largest_face)



    gridX,gridY = create_grid(largest_face, image.shape[1], image.shape[0])
    gridX = np.expand_dims(gridX, axis=(0, -1))*128
    gridY = np.expand_dims(gridY, axis=(0, -1))*128
    gridX = np.floor(gridX).astype(np.int8)
    gridY = np.floor(gridY).astype(np.int8)
    return face, gridX,gridY, debug_image

def convert_and_resize_image(image, resize_size, gray = True):
    # Resize the image using OpenCV
    resized_image = cv2.resize(image, resize_size)
    if gray:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Convert the resized image to numpy array
    resized_np = (np.array(resized_image).astype(np.float32)-127.5).astype(np.int8)
    if gray:
        return np.expand_dims(resized_np, axis=[0,-1])
    else:
        return np.expand_dims(resized_np, axis=[0])

def draw_point_on_image(image, x_offset, y_offset,color = (0, 0, 255) ):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the middle point of the image
    center_x = width // 2
    center_y = height // 2

    # Calculate the coordinates of the point based on the offsets
    point_x = int(center_x - x_offset)
    point_y = int(center_y - y_offset)

    # Draw a point on the image using a circle
    radius = 6
    thickness = -1  # Fill the circle
    cv2.circle(image, (point_x, point_y), radius, color, thickness)

    # Return the image with the drawn point
    return image

def draw_cross_on_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the middle point of the image
    center_x = width // 2
    center_y = height // 2

    # Set the cross line properties
    line_color = (100, 100, 100,100)  # Red color
    line_thickness = 2

    # Draw the vertical line
    cv2.line(image, (center_x, 0), (center_x, height), line_color, line_thickness)

    # Draw the horizontal line
    cv2.line(image, (0, center_y), (width, center_y), line_color, line_thickness)

    # Return the image with the cross
    return image


tinytracker_interpreter = tf.lite.Interpreter(model_path="models/TinyTracker.tflite")
tinytrackerS_interpreter = tf.lite.Interpreter(model_path="models/TinyTrackerS.tflite")

tinytracker_interpreter.allocate_tensors()
tinytrackerS_interpreter.allocate_tensors()

input_details = tinytracker_interpreter.get_input_details()
input_details_S = tinytrackerS_interpreter.get_input_details()


input_shape_tinytracker = input_details[0]["shape"]
input_shape_tinytrackerS = input_details_S[0]["shape"]


cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open webcam")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ps = [0,0]

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #frame = crop_image(frame, (800,0,720,1000))

    # Check if frame is successfully read
    if not ret:
        print("Failed to read frame")
        break

    # Display the frame in a window named "Webcam"
    try:
        face, gridX, gridY, image = detect_eyes_and_face(frame)

    except:
        image = frame
        face = None
    image = draw_cross_on_image(image)
    if face is not None:
        face_with_grid = np.concatenate([(convert_and_resize_image(face, input_shape_tinytracker[1:3], gray=False).astype('int32')).astype('int8'), gridX, gridY], axis=-1)
        face = (convert_and_resize_image(face, input_shape_tinytrackerS[1:3]).astype('int32')).astype('int8')

        tinytracker_interpreter.set_tensor(input_details[0]['index'], face_with_grid)
        tinytrackerS_interpreter.set_tensor(input_details_S[0]['index'], face)

        tinytracker_interpreter.invoke()
        tinytrackerS_interpreter.invoke()

        output_details = tinytracker_interpreter.get_output_details()
        output_details_s = tinytrackerS_interpreter.get_output_details()

        pred = tinytracker_interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
        pred_S = tinytrackerS_interpreter.get_tensor(output_details_s[0]['index']).astype(np.float32)
        #b = b.cpu().detach().numpy()
        print("TinyTrackerS:", pred)
        print("TinyTracker:", pred_S)
#           if b >= 0.5:
#                 print("blink")
        image = draw_point_on_image(image, pred[0][0] * 540, pred[0][1] * 540)
        image = draw_point_on_image(image, pred_S[0][0] * 540 // 3, pred_S[0][1] * 540 // 3, color=(255, 0, 255))


    #image = draw_point_on_image(image, ps[0] * 100, ps[1] * 100)
    cv2.imshow("Webcam", image)

    # Wait for the 'q' key to be pressed to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

    # Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()