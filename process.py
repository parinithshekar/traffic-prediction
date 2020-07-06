# Standard library
import time
import json

# Packages
import cv2
import numpy as np

# Custom
from load import traffic_model

model = traffic_model()

''' Refer to the paper for detailed explanation of hyper-parameters and algorithm '''

# Hyper-paramenter depending on the speed, size of the vehicle
# and the size of the road (in secs)
WEIGHTS =[0, 5, 8, 0, 3, 2, 7]

# OPTIMAL PROCESSING TIME - Pillar of the algorithm
# Hyper-parameter depending on the horizontal distance b/w vehicles and camera,
# camera's angle and its position (per vehicle class)
OPT = [1, 3.971544, 4, 4, 4, 4, 4]

# Lower and upper limits for the signal's green time
MIN_TIME = 15
MAX_TIME = 70

# We are using SSD 300
IMG_WIDTH = 300
IMG_HEIGHT = 300

# Classes in the last layer of model. DO NOT change order.
classes = [ 'background', 'car', 'truck', 'person', 'bicycle', 'motorcycle', 'bus']
NUM_OBJECTS = len(classes)

# Algorithm calculation
green_time = MIN_TIME
prev_predict = 0
present_approx = 0
next_predict = 0
alpha = 0.5

# Flags indicating start and end of green signal
open_flag = 1
close_flag = 0

# Frame processing parameters
confidence_threshold = 0.5
frame_vehicle_count = [0] * NUM_OBJECTS
total_vehicle_count = [0] * NUM_OBJECTS
vehicles = { 'background':0, 'car':0, 'truck':0, 'person':0, 'bicycle':0, 'motorcycle':0, 'bus':0 }

# Set video source
capture = cv2.VideoCapture("cctv_crop.mp4")
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

# Set the colors for the bounding boxes
colors = [tuple(255 * np.random.rand(3)) for i in range(NUM_OBJECTS)]


# Continuous video feed
while(True):

    # ret is True if read is successful
    ret, frame = capture.read()
    if (not ret):
        break
    else:
        #start time of frame processing
        frame_start_time = time.time()
        
        # Resize frame for model
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame = np.array(frame)
        model_input_frame = np.expand_dims(frame, axis=0)

        # Get model predictions - vehicles detected
        y_pred = model.predict(model_input_frame)
        
        # Filter detected vehicles with confidence threshold
        y_pred_decoded = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        '''
        # Print raw output of the model prediction
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('    class    conf  xmin     ymin   xmax    ymax')
        print(y_pred_decoded[0])
        '''
        
        # Signifies cumulative weight of all vehicles in the given frame
        frame_weight = [0] * NUM_OBJECTS

        # Count vehicles in the current frame starting from zero
        frame_vehicle_count = [0] * NUM_OBJECTS

        # For every detected vehicle
        for box in y_pred_decoded[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            vehicle_index = int(box[-6])

            # Coordinates for box around detected vehicle
            xmin = int(box[-4] * frame.shape[1] / IMG_WIDTH)
            ymin = int(box[-3] * frame.shape[0] / IMG_HEIGHT)
            xmax = int(box[-2] * frame.shape[1] / IMG_WIDTH)
            ymax = int(box[-1] * frame.shape[0] / IMG_HEIGHT)

            # Draw labelled box around detected vehicle
            color = colors[vehicle_index]
            label = classes[int(box[0])]
            confidence = "{:.2f}".format(box[1])
            processed_frame = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, 2)
            processed_frame = cv2.putText(processed_frame, label, (xmin+2,ymin+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            processed_frame = cv2.putText(processed_frame, confidence, (xmin+2,ymax-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            # Accumulate the weight corresponding to every vehicle detected
            frame_weight[vehicle_index] += WEIGHTS[vehicle_index]

            # Increment count of the detected vehicle
            frame_vehicle_count[vehicle_index] += 1

        # Time taken to process this frame (in secs)
        frame_processing_time = time.time() - frame_start_time

        # Present frame's contribution towards detecting vehicles in OPT
        # Calculated per vehicel class
        frame_contribution = [(frame_processing_time/vehicle_opt) for vehicle_opt in OPT]
        # frame_contribution = frame_processing_time / OPT

        print(frame_weight)
        
        # Executed when the signal is green (open)
        if (open_flag==1 and close_flag==0):
            
            # For every vehicle type
            for vehicle_index in range(NUM_OBJECTS):
                # Add this frame's contribution to the count
                present_approx += (frame_weight[vehicle_index] * frame_contribution[vehicle_index])
                print(vehicle_index)
                # Add the frame_contribution weighted count to get approximate vehicles counted
                total_vehicle_count[vehicle_index] += (frame_vehicle_count[vehicle_index] * frame_contribution[vehicle_index])

                vehicle_class = classes[vehicle_index]
                vehicles[vehicle_class] = total_vehicle_count[vehicle_index]

        # Executed when the signal becomes red (closed), denoted by close_flag
        if (open_flag==0 and close_flag==1):
            
            prev_predict = next_predict
            # Shortest Job First # OS_Component
            # Weighted average of previous and current estimate for green time
            next_predict = (present_approx * alpha) + ((1-alpha) * prev_predict)

            # The bounds of the green signal time = min < prediction < max
            if ( (MIN_TIME < next_predict) and (MAX_TIME > next_predict) ):
                green_time = next_predict
            elif (MIN_TIME > next_predict):
                green_time = MIN_TIME
            else:
                # MAX_TIME < next_predict
                green_time = MAX_TIME

            present_approx = 0
            close_flag = 0

        # Print output to terminal
        print(f'Next Prediction {green_time}')
        print(f'Start_Flag {open_flag}')
        print(f'End_Flag {close_flag}')

        fps = 1 / frame_processing_time
        print(f'FPS {fps}')

        print("\nVehicles counted:\n")
        print(json.dumps(vehicles, indent=2))

        # Display processed frame
        cv2.imshow('BroViolinBro', processed_frame)
        
        # Take keypress as input
        key = cv2.waitKey(1) & 0xFF
        # Simulate opening of signal (green)
        if key == ord('s'):
            open_flag = 1
            close_flag = 0
        # Simulate closing of signal (red)
        elif key == ord('e'):
            open_flag = 0
            close_flag = 1
    
        # Close program
        elif key == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()