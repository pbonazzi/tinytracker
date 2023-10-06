import serial
import cv2
import numpy as np
from datetime import datetime
def plot_points(points,points_com, window_size=(1080, 1920,3)):
    # Create a white background
    img = np.ones(window_size, dtype=np.uint8) * 255

    # Calculate the center of the image
    center = (window_size[0] // 2, window_size[1] // 2)

    # Draw a cross through the center of the image
    cv2.line(img, (center[1], 0), (center[1], window_size[0]), (0, 0, 0), 1)
    cv2.line(img, (0, center[0]), (window_size[1], center[0]), (0, 0, 0), 1)

    # Plot points
    for point in points:
        # Convert coordinates to match image space
        img_point = (center[1] - int(point[1]), center[0] - int(point[0]))

        # Draw the point
        cv2.circle(img, img_point, radius=4, color=(255, 0, 0), thickness=-1)


    # Show the image
    cv2.imshow("Points", img)
    if cv2.waitKey(1) == 'q':
        cv2.destroyAllWindows()
# Configure the serial port
port = '/dev/ttyUSB0'  # Replace with your serial port
baudrate = 115200
points = []
points_com = []
# Create a serial object
ser = serial.Serial(port, baudrate)

window_size = (540,960,3)

#tinytracker_interpreter = tf.lite.Interpreter(model_path="itrackerpro/finetuned_best.tflite")
#tinytracker_interpreter.allocate_tensors()
#input_details = tinytracker_interpreter.get_input_details()


#output_details = tinytracker_interpreter.get_output_details()

print("start demo")
time_stamp = datetime.now()
# Read from the serial port
while True:
    line = ser.readline().decode().strip()  # Read a line from the serial port and decode it
    print(line)
    # Check if the line matches the expected format
    if line.startswith('face_with_grid:') and 'y:' in line:
        try:
            print("Time:", datetime.now()- time_stamp)
            # Extract the face_with_grid and y values
            vals = line.split()
            y_value = float(vals[0][2:])
            x_value = float(vals[2])

            # Print the extracted values
            print('face_with_grid:', x_value)
            print('y:', y_value)
            points.append([y_value*window_size[0]//2,x_value*window_size[1]//2])
            plot_points(points[-4:], points_com[-4:], window_size)
            time_stamp = datetime.now()
        except ValueError:
            print('Error parsing floating-point numbers')
    elif line.startswith("Nutt"):
        ser.write(b"tf_example\n")
    elif line.startswith("START"):
        image = []
        line = ser.readline().decode().strip()  # Read a line from the serial port and decode it
        while not line.startswith("END"):
            p = int(line)
            image.append(p)
            line = ser.readline().decode().strip()  # Read a line from the serial port and decode it
        print(image)
        image_data = np.array(image).reshape(1,96, 96,1)
        image_data = np.uint8(image_data)
        cv2.imshow("img", image_data[0,:,:,0])
        #tinytracker_interpreter.set_tensor(input_details[0]['index'],image_data.astype('int8'))
        #tinytracker_interpreter.invoke()
        #pred = tinytracker_interpreter.get_tensor(output_details[0]['index']).astype(np.float32)[0]
        #points_com.append([-pred[0]*250,pred[1]*250])
        #b = b.cpu().detach().numpy()
        print(p)
        if cv2.waitKey(1) == 'q':
            cv2.destroyAllWindows()

# Close the serial port
ser.close()