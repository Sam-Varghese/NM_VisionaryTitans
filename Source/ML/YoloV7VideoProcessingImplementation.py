# Yolov7 video processing format

import cv2
import time
import random
import numpy as np
import onnxruntime as ort
import pyttsx3
import multiprocessing
import math
import mysql.connector
from mysql.connector import errorcode
import uuid

# Class for dealing with database connections
class DatabaseConnector:
    """For interacting with MySQL database."""
    def __init__(self):
        self.username = "root"
        self.password = "root"
        self.host = "localhost"
        self.database = "NM_VisionaryTitans"
        self.connection = None

    def connect(self):
        """Establishes python and MySQL connection, creates the database if it doesn't exist."""
        try:
            # Establish a connection to the MySQL server
            self.connection = mysql.connector.connect(
                user=self.username,
                password=self.password,
                host=self.host,
            )
            self.cursor = self.connection.cursor()

            self.cursor.execute("CREATE DATABASE IF NOT EXISTS {}".format(self.database))
            self.connection.commit()

            self.cursor.execute("USE {}".format(self.database))

            print("Connected to the database.")

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Error: Access denied. Check your username and password.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Error: The specified database does not exist.")
            else:
                print("An error occurred:", err)

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Disconnected from the database.")

    def create_tables(self):
        try:

            # Define the table creation statement
            create_table_query = """CREATE TABLE IF NOT EXISTS realTimeTrends (
                  id VARCHAR(255) PRIMARY KEY,
                  StartTime VARCHAR(255) NOT NULL,
                  EndTime VARCHAR(255) NOT NULL,
                  PeopleCount INT,
                  VehicleCount INT,
                  AverageSpeed FLOAT NULL
                )"""

            # Execute the table creation statement
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("Tables created successfully.")


        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateRealTimeTrends(self, start_time, end_time, people_count, vehicle_count, avg_speed):
        try:
            if(avg_speed == None):
                avg_speed = "NULL"
            # Define the INSERT statement
            insert_query = """INSERT INTO realTimeTrends 
                (Id, StartTime, EndTime, PeopleCount, VehicleCount, AverageSpeed) 
                VALUES ('{}', '{}', '{}', {}, {}, {})""".format(str(uuid.uuid4()), start_time, end_time, people_count, vehicle_count, avg_speed)
            # print("Executing the query\n", insert_query)
            
            self.cursor.execute(insert_query)
            self.connection.commit()

        except mysql.connector.Error as err:
            print("An error occurred:", err)

engine = pyttsx3.init()
databaseConnector = DatabaseConnector()
databaseConnector.connect()
databaseConnector.create_tables()

# Getting video inputs
# video_path = input("Enter the path of video to analyze: ")
video_path = "rough/accidents/cyberabad_traffic_incident1.mp4"
video_capture = cv2.VideoCapture(video_path)

# Getting the video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = video_path.replace(".mp4", "_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Imported necessary libraries...")

cuda = True # For utilizing GPU, and performing parallel computing
weights = "Source/ML/yolov7-tiny.onnx"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
# Creating an inference session to utilize the pre build model
session = ort.InferenceSession(weights, providers=providers)

all_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']

# Target classes on which I need to focus
classes = ["person", "bus", "bicycle", "car", "truck", "motorcycle"]
# classes = all_classes

# Generating random colors or bounding box of each of these classes

colors = {}

for class_name in classes:

    colors[class_name] = tuple([random.randint(0, 255) for _ in range(3)]) # Generating (r, g, b) list

print("Generated random colors for object bounding boxes")

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]
thickness = 2

class Object:
    """Creating class to store details of each and every detected object that gets detected by Yolov7"""
    def __init__(self, object_type, class_id, bbox_coordinates):
        self.id = object_type + str(random.randint(0, 100)) + chr(random.randint(97, 122)) # Not unique, but mostly unique
        self.class_id = class_id
        self.bbox = bbox_coordinates # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        self.failure_count = 0 # count of failure in detecting a worthy child
        self.time_bbox_updates = {} # Format: {0: [bbox, time_stamp], 1: [bbox, time_stamp]}, used for calculating speed and trajectory of object

    def find_worthy_child(self, new_bbox_list):
        min_dist = abs(self.bbox[2] - self.bbox[0])/2
        min_dist = 100
        selected_element_index = None
        counter = 0
        for new_bbox in new_bbox_list:
            dist = math.dist(self.bbox[:2], new_bbox[:2])
            if(dist <= min_dist):
                # Detected worthy child
                min_dist = dist
                worthy_child_bbox = new_bbox
                selected_element_index = counter

            counter += 1
        # If no bbox has been found as a worthy child
        if(selected_element_index == None):
            self.failure_count += 1
            return None
        else:
            # Reset the failure counter
            self.failure_count  = 0

        # If a bbox is found worthy
        self.bbox = worthy_child_bbox[:]
        
        return selected_element_index

# For OpenCV fonts
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.0

# Calculating time taken to process each frame
start_time = time.time()
img_counter = 1

# Variable to check if video processing is still on
video_processor_active = True

# OpenCV camera capture
def start_ai_cam(objects_detected, video_processor_active):
    try:

        # Starting OpenCV Video Capture
        print("Initiating camera...")
        
        # Create a named window for full screen display
        cv2.namedWindow("Live Footage", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Live Footage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Setting frame as a global variable to make it accessible to multiple processors 
        img_counter = 1

        while video_capture.isOpened():

            ret, frame = video_capture.read()
            if not ret:
                end_time = time.time()
                print("Avg frame processing time (time taken/ frames processed): ",(end_time- start_time)/img_counter, " ie FPS=", img_counter/(end_time-start_time))
                print("Total time taken to analyze the video: ", (end_time - start_time))
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLO requires BGR images to be in RGB form

            resize_data = []

            image_cpy, ratio, dwdh = letterbox(image, auto=False)

            image_cpy = image_cpy.transpose((2, 0, 1))
            image_cpy = np.expand_dims(image_cpy, 0) # Adds an extra dimension to image at index 0, YOLOv7 format
            image_cpy = np.ascontiguousarray(image_cpy) # Changes the matrix structure of image_cpy, again YOLOv7 format
            image_cpy = image_cpy.astype(np.float32)

            resize_data.append((image_cpy, ratio, dwdh))

            # Running batch 1 inference
            # We can speed up the video processing by opting batch 32, and 64 inference, butit's quite difficult and logically incorrect to implement it, as for 32 inference, it should wait for 32 frames to get captures, only then it'll give it's prediction.

            image = np.ascontiguousarray(resize_data[0][0]/255) # Normalizing the image
            prediction = session.run(outname, {"images": image})

            enu_predic = enumerate(prediction)

            new_bbox = {}

            for i, prediction_array in enu_predic:
                
                # If n objects gets detected in the frame, then prediction_array will contain n arrays inside it containing batch_id, x0, y0, x1, y1, cls_id, score

                for (batch_id, x0, y0, x1, y1, cls_id, score) in prediction_array:
                    # Coordinates are of top left and bottom right

                    # Allows only those detections to be shown as output whose confidence value is above a threshold
                    if score < 0.2: # The confidence value of predicted object
                        continue
                    
                    cls_id = int(cls_id)
                    class_name = all_classes[cls_id]

                    if class_name in classes:

                        # Reversing the paddings and other transformations applied during letterbox

                        box = np.array([x0,y0,x1,y1])
                        box -= np.array(dwdh*2)
                        box /= ratio
                        box = box.round().astype(np.int32).tolist()
                        try:
                            new_bbox[cls_id].append(box)
                        except Exception:
                            new_bbox[cls_id]=[box]
            for class_id in list(new_bbox):
                class_name = all_classes[class_id]
                class_color = colors[class_name]
                # Allowing only the class_id objects to find their worthy child
                
                try:
                    obj_of_id = objects_detected[class_id] # needs to be reversed
                    for obj in obj_of_id: # objects_detected is created in the except block first
                        worthy_status = obj.find_worthy_child(new_bbox[class_id])
                        # If a bbox is found worthy
                        if(worthy_status != None):
                            cv2.rectangle(frame, obj.bbox[:2], obj.bbox[2:], class_color, thickness)
                            cv2.putText(frame, obj.id, obj.bbox[:2], font, font_scale, class_color, thickness)
                            # Pop out the bbox from new_bbox as it's father has been detected
                            try:
                                new_bbox[class_id].pop(worthy_status)
                            except Exception:
                                # In the situation where there's no key like class_id in new_bbox
                                pass
                        elif (worthy_status == None and obj.failure_count >=3):
                            obj_of_id.remove(obj) # If the object didn't find any successor for 3 times in a row, then let's drop the object so tracker won't be looking for it against other bboxes all the time. reduces load significantly by preventing accumulation of instances that are no longer in the screen

                except Exception as e:
                    # Will be executed if the class_id key doesn't exist in the dict objects
                    obj_of_id = []

                # Creating new objects for new bboxes discovered that didn't prove as a worthy child of any existing object
                for unused_bbox in new_bbox[class_id]:
                    # Create objects of only those types, which are in the list of classes to get detected
                    if(class_name not in classes):
                        continue

                    obj = Object(class_name, class_id, unused_bbox)

                    try:
                        obj_of_id.append(obj)
                    except Exception:
                        obj_of_id = [obj]

                    cv2.rectangle(frame, obj.bbox[:2], obj.bbox[2:], class_color, thickness) # Passing coordinates of top left and bottom right
                    cv2.putText(frame, obj.id, obj.bbox[:2], font, font_scale, class_color, thickness)

                objects_detected[class_id] = obj_of_id # This is the method to update a special multi processor variable

            # Saving the annotated frames
            output_video.write(frame)
            cv2.imshow("Live Footage", frame)

            # Used to calculate the FPS of the program
            img_counter += 1

            if (cv2.waitKey(1) == ord("q")):
                end_time = time.time()
                print("Avg frame processing time (time taken/ frames processed): ",(end_time- start_time)/img_counter, " ie FPS=", img_counter/(end_time-start_time))
                break


    except Exception as e:
        import traceback
        # print("Encountered an unexpected exception {}".format(e))
        print(traceback.print_exc())
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        print(f"Exception occurred at line {line_number}")

    finally:

        video_processor_active.value = False # This also terminates multi processor activities, enabling clean exit
        video_capture.release()
        output_video.release()
        cv2.destroyAllWindows()

def get_objects_count(objects_detected):
    """Returns the count of people and vehicles detected, assuming that there are only 2 categories that are being detected by YOLOv7, ie people, and vehicles (car, truck, etc)"""
    persons_count = 0
    vehicle_count = 0
    # print("LIst of objects detected: ", list(objects_detected))

    for class_id in list(objects_detected.keys()):
        if(class_id == 0):
            persons_count = len(objects_detected[class_id])
        else:
            vehicle_count += len(objects_detected[class_id])
    print("Counts are : ",  [persons_count, vehicle_count])
    return [persons_count, vehicle_count]

def calculate_speed(objects_detected):
    """This function needs to run twice in order to determine the average speed of the vehicle. Make sure to reset object.time_bbox_updates to {} after running this function twice"""
    previous_avg_speeds = []
    objects_detected_ids = [i for i in list(objects_detected.keys()) if i != 0]
    for id in objects_detected_ids:
        obj_of_id = objects_detected[id]
        
        for object in obj_of_id:
            print("Detected object of type {}".format(object.class_id))
            if(object.class_id != 0): # If the object's not a person
                
                obj_len = len(object.time_bbox_updates) # Putting the obj_len'th observation into object.time_bbox_updates
                
                if (obj_len < 2):
                    object.time_bbox_updates[obj_len] = [object.bbox, time.time()]

                else:
                    obj_dist = math.dist(object.time_bbox_updates[0][0][:2], object.time_bbox_updates[1][0][:2]) # Calculating distance between top left coordinates of same object, at different time intervals
                    obj_time = object.time_bbox_updates[1][1] - object.time_bbox_updates[0][1]
                    previous_avg_speeds.append(obj_dist/obj_time)
        objects_detected[id] = obj_of_id
        return previous_avg_speeds # the value returned when this function is executed twice is the final value

def reinitialize_time_bbox_updates(objects_detected):
    """Re-initializes the time bbox updates in order to prevent un-useful data accumulation while executing real time general data collector. Execute this after calculate_speed is run twice."""
    objects_detected_keys = list(objects_detected.keys())
    for id in objects_detected_keys:
        objects_of_id = objects_detected[id]
        for object in objects_of_id:
            object.time_bbox_updates = {}
        objects_detected[id] = objects_of_id

def realTimeGeneralDataCollector(objects_detected, video_processor_active):
    """Records the average speed of vehicles present in the time duration of 2 seconds, along with the count of vehicles and people."""
    # time.sleep(5) # In order to let the frames get captured, and some processing done when the program is run first

    # Getting all the data
    start_time = time.strftime("%d %B, %Y %I:%M %p", time.localtime(time.time()))
    # print("Objects detected: ", objects_detected)
    objects_count = get_objects_count(objects_detected)
    calculate_speed(objects_detected)

    time.sleep(0.5) # Let the objects detected move a bit to analyze their avg speeds
    calculate_speed(objects_detected)

    avg_speeds = calculate_speed(objects_detected) # Needs to run twice
    if (avg_speeds == []) or (avg_speeds == None):
        avg_speed = None
        print("No vehicles detected")
    else:
        avg_speed = sum(avg_speeds)/len(avg_speeds)
    reinitialize_time_bbox_updates(objects_detected) # To prevent accumulation of data and enable expected functioning of calculate_speed
    end_time = time.strftime("%d %B, %Y %I:%M %p", time.localtime(time.time()))
    print("Attempting database entry...")
    databaseConnector.updateRealTimeTrends(start_time, end_time, objects_count[0], objects_count[1], avg_speed)

    # time.sleep(2) # Putting a random sleep statement just to start recording next set of data after some time, ie. it'll record status in approx every 5 seconds. It's safe to remove this, but a lot of data will get generated
    print("Data saved")
    print(video_processor_active.value, type(video_processor_active.value))
    if (video_processor_active.value): # If the video is also getting processed simultaneously, terminate the recursion
        realTimeGeneralDataCollector(objects_detected, video_processor_active)

def text_to_speech(text: str):

    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":

    # Creating multiple processes
    # Performing a lot of machine learning computations can be too expensive, hence it's better to split it up among multiple processors to ensure a distributed, faster work

    # Creating shared variables among multi processors because even though a variable is made global, it doesn't update for all processor tasks
    manager = multiprocessing.Manager()

    objects_detected = manager.dict()
    video_processor_active = multiprocessing.Value("b", True)
    print(video_processor_active.value)
    live_yolo_detection_process = multiprocessing.Process(target = start_ai_cam, args = (objects_detected, video_processor_active))
    realTimeGenDataCollector = multiprocessing.Process(target = realTimeGeneralDataCollector, args = (objects_detected, video_processor_active))

    # Starting both the processes at the same time

    live_yolo_detection_process.start()
    realTimeGenDataCollector.start()

    # Waiting until both the processes gets finished
    live_yolo_detection_process.join()
    realTimeGenDataCollector.join()

    databaseConnector.disconnect()

    print("Clean exited all tasks being executed by multiple processors")