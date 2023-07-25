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
from rich.console import Console
# Anomaly detection algo class
import scipy.stats as stats
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

console = Console()
# Class for dealing with database connections
class DatabaseConnector:
    """For interacting with MySQL database."""
    def __init__(self):
        self.username = "root"
        self.password = "root"
        self.host = "localhost"
        self.database = "NM_VisionaryTitans"
        self.connection = None
        self.gen_rows_inserted = 0
        self.sp_rows_inserted = 0
        self.generalInfoTable = None
        self.specificInfoTable = None

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
            create_table1_query = """CREATE TABLE IF NOT EXISTS {} (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  StartTime VARCHAR(255) NOT NULL,
                  EndTime VARCHAR(255) NOT NULL,
                  PeopleCount INT,
                  VehicleCount INT,
                  AverageSpeed FLOAT NULL
                )""".format(self.generalInfoTable)
            
            create_table2_query = """CREATE TABLE IF NOT EXISTS {} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                time VARCHAR(255) NOT NULL,
                vehicleName VARCHAR(255) NOT NULL,
                topLeftX FLOAT NOT NULL,
                topLeftY FLOAT NOT NULL,
                bottomRightX FLOAT NOT NULL,
                bottomRightY FLOAT NOT NULL,
                speed FLOAT NULL
            )""".format(self.specificInfoTable)

            # Execute the table creation statements
            self.cursor.execute(create_table1_query)
            self.cursor.execute(create_table2_query)

            self.connection.commit()
            print("Tables created successfully.")


        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateGenTable(self, start_time, end_time, people_count, vehicle_count, avg_speed):
        try:
            if(avg_speed == None):
                avg_speed = "NULL"
            # Define the INSERT statement, here UUID has been used because nor vehicle/ person't ID will be suitable to define a particular time instant
            insert_query = """INSERT INTO {} 
                (StartTime, EndTime, PeopleCount, VehicleCount, AverageSpeed) 
                VALUES ('{}', '{}', {}, {}, {})""".format(self.generalInfoTable, start_time, end_time, people_count, vehicle_count, avg_speed)
            
            self.cursor.execute(insert_query)
            self.connection.commit()
            self.gen_rows_inserted += 1
            # print("Inserted {}th datapoint of general data".format(self.gen_rows_inserted))

        except mysql.connector.Error as err:
            print("An error occurred:", err)

    def updateSpTable(self, time, vehicle_name, topLeftX, topLeftY, bottomRightX, bottomRightY, speed):
        try:
            insert_query = """INSERT INTO {} 
                (time, vehicleName, topLeftX, topLeftY, bottomRightX, bottomRightY, speed) 
                VALUES ('{}', '{}', {}, {}, {}, {}, {})""".format(self.specificInfoTable, time, vehicle_name, topLeftX, topLeftY, bottomRightX, bottomRightY, speed)
            
            self.cursor.execute(insert_query)
            self.connection.commit()
            self.sp_rows_inserted += 1
            # print("Inserted {}th datapoint of specific data".format(self.sp_rows_inserted))

        except mysql.connector.Error as err:
            print("An error occurred at function updateSpTable:", err)

    def clearTables(self):
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(self.generalInfoTable))
        self.cursor.execute("DROP TABLE IF EXISTS {}".format(self.specificInfoTable))

        self.connection.commit()
        print("All tables cleared.")

# Class for performing anomaly detection
class AnomalousMeans:
    """For analyzing anomalous means if they appear in our set using ANOVA (f test). This program should be run after a considerable amount of time (for collecting all data points)."""
    def __init__(self, alertsInstance):
        self.checkpoints = []
        self.newCheckpointBuffer = [] # It accumulates checkpoint data before finally sending it to checkpoints
        self.threshold_value = 0.05
        self.alertsInstance = alertsInstance # Set this manually as this instance will be used to throw alerts
        self.lastCheckpointTime = time.time() # To keep a track of time when the data and anomalous checks were last performed
        self.anomalyCheckTime = 7 # Number of seconds after which data should get checked for anomaly
        self.checkpointLimit = 5 #Number of checkpoint to be gathered for anomaly detection every time
        self.detectionField = "None"

    def autoTrainerChecker(self, checkpointData: float):
        """This function is for either adding the data, or checking for anomalous situations based on whether it has collected enough data, or does not have data for performing anomaly mean detection. These checks should happen automatically. Argument taken is a single number might be a speed, etc."""
        print("Trying to check anomalous means")
        if((time.time() - self.lastCheckpointTime) >= self.anomalyCheckTime): # If it's the time to add checkpoints and check for anomaly
            self.add_checkpoint(self.newCheckpointBuffer) # Data not getting updated error
            self.newCheckpointBuffer = [] # Reset it's value
            if (len(self.checkpoints) > 3):
                self.checkMeanAnomaly()
                # self.checkpoints.pop(0) # To avoid memory leaks
        else: # Else just keep adding data to buffer
            console.print("Time limit does not approve anomalous mean detection {} sec. last checkpoint time = {}, and current time = {}".format(time.time() - self.lastCheckpointTime, self.lastCheckpointTime, time.time()), style = "red")
            if(checkpointData != None):
                self.newCheckpointBuffer.append(checkpointData)

    def add_checkpoint(self, new_checkpoint):
        """In order to compare means, it stores the new checkpoint values. Works intelligently in a way that it automatically discards older checkpoints, replaces them with newer ones."""
        if(len(self.checkpoints) >= self.checkpointLimit):
            # Remove 0th element, add 1 list to the end of self.checkpoints
            self.checkpoints.pop(0)

        # Simply add the checkpoint if the len is not checkpointLimit yet
        self.checkpoints.append(new_checkpoint)
        print("Updating the last checkpoint time...")
        self.lastCheckpointTime = time.time()

    def checkMeanAnomaly(self):
        # print(self.checkpoints)
        f_statistics, p_value = stats.f_oneway(*self.checkpoints)
        console.print("Checking mean anomaly...", style = "green")
        if(p_value < self.threshold_value):
            # Give an alert because the means differ significantly
            self.alertsInstance.initiateMediumLevelAlert("Means differ significantly in {}.".format(self.detectionField))
            console.print("Debug info: \ncheckpoints list = ", self.checkpoints)
            # text_to_speech("Significant mean difference in {}".format(self.detectionField))
            # print("Checkpoints are ", self.checkpoints)

            return True
        else:
            console.print("Means don't differ significantly", style = "green")
            console.print("Debug info: \ncheckpoints list = ", self.checkpoints)
            return False

class OutlierDetector:
    """Detects outliers using Local Outlier Factory Method. Applicable for outlying coordinates and values like speeds, etc. Made to keep a continuous check on the incoming real time data."""
    def __init__(self):
        self.data = np.array([])
        self.neighbours_count = 3 # Keep it low to make it more sensitive to all variations. Keeping it high lets it ignore certain variations, and only catch those who are just too much anomalous
        self.alertsInstance = None # Set this manually as this instance will be used to throw alerts
        self.dataLimit = 600 # Increase it to make this anomaly detection algo work more accurately, at the cost of increased memory load on embedded device
        self.anomalyDetectionStartThreshold = 300 # After we get this much data, we can start applying anomaly detection. The max permissible data will still be self.dataLimit
        self.anomalyThreshold = 2
        self.detectionField = "Field"

    def autoOutlierUpdaterAnomalyChecker(self, data_point):
        """Data point should be np array, either a single number or 2 numbers indicating x and y coordinates."""
        return_val = False
        if(len(self.data) >= self.anomalyDetectionStartThreshold): # Else if the count of data items is atleast above the threshold value to start it, simply add the data and perform anomaly detection simultaneously
            self.addTrainingData(data_point)
            if(self.detectOutliers()):
                # text_to_speech("Got enough data to start outlier detection")
                return_val = True
        else:
            # console.print("Not enough values to start outlier detection for ",self.detectionField , ". Currently I've ", len(self.data), " not 500", style = "red")
            pass
            # print(self.data)
        if(len(self.data)>=self.dataLimit):# If the data limit already exceeded
            self.data = self.data[1:] # Keep removing elements from beginning, and appending new ones at the end. This might affect the performance a little, but for now, it won't hamper much
            # console.print("Removed an ele from outlier detection to avoid mem leak", style = "yellow")
            self.addTrainingData(data_point)
        else:
            # print("Trying to data of {} {}".format(self.detectionField, data_point))
            if(data_point[0] != None):
                self.addTrainingData(data_point)
        return return_val

    def addTrainingData(self, data_point):
        """In order to make this algorithm work, it needs to get all the possible usual data, and then it would be able to detect any possible outliers. Remember to update this data as per current trends, so the ML model would get the feature of adapting with latest trends. For updating, all you need to do is to keep adding more and more data. Don't worry about the previous data dominating over newer data. Data point should be np array, either a single number or 2 numbers indicating x and y coordinates."""
        print("Data point recieved for field {} is {}".format(self.detectionField, data_point))
        if(np.any(data_point)): # If data point is not empty
            print("Entered if")
            data_point = data_point.reshape(1, -1)
            # console.print("Incoming data of {} is not none.... {}, field: {}".format(self.detectionField, data_point, self.detectionField))
            # print("Incoming data: ", data_point)
            # print("Current dataset: ", self.data)
            if (np.any(self.data)): # For appending data though np.vstack, it should initially have some element, shouldn't be empty
                # self.data = data_point
                try:
                    # console.print("apply vstack: data = {} and data point = {} for field {}".format(self.data, data_point, self.detectionField), style = "cyan")
                    self.data = np.vstack((self.data, data_point))
                except Exception as e:
                    console.print("Exception occured while applying vstack",style = "red")
                # console.print("data is not none")
            else:
                self.data = data_point
                # console.print("Data is none for field {}".format(self.detectionField), style = "red")
        else:
            # console.print("Data point is none.... for field {} {} expression comparision to None {}".format(self.detectionField, data_point, (data_point == [])), style = "red")
            pass

    def detectOutliers(self):
        console.print("Detecting outliers", style = "green")
        lof = LocalOutlierFactor(n_neighbors=self.neighbours_count)
        lof_scores = lof.fit_predict(self.data)
        lof_scores = -lof.negative_outlier_factor_  # Give a quantity of how much that particular point is outlying, so it also contains the same number of elements as in self.data
        # print("Current LOF scores = ", lof_scores, " min lof = ", np.min(lof_scores), " max lof = ", np.max(lof_scores), " mean lof = ", np.mean(lof_scores))
        threshold = np.mean(lof_scores)*self.anomalyThreshold
        outliers = self.data[np.logical_or(lof_scores > threshold, lof_scores < threshold*(-1))]
        if(len(outliers) != 0):
            self.alertsInstance.initiateMediumLevelAlert("Detected Outliers in {}.".format(self.detectionField))
            text_to_speech("Detected outlier in {}".format(self.detectionField))
            print("Data of outlier is ", self.data)
            # if(self.detectionField == "Speed"):
                # print("Data of outlier is ", self.data)
            return True
        else:
            return False
        # Now send an alert, the level of alert can be found through getting it's lof_score

class DBSCAN:
    def __init__(self, min_radius, min_pts, alert_system):
        self.coordinates = None
        self.min_radius = min_radius # Set this to adjust sensitivity towards distance
        self.min_pts = min_pts # Minimum points to come inside the circle in order to call it a core point
        self.direct_density_reachable_count = 0
        self.core_points_count = 0
        self.core_pt_threshold = 5 # Set this to adjust sensitivity towards crowd
        self.alert_system = alert_system
        self.single_time_runner = 0 # Just a variable to use in order to make a function run only once

    def findClusters(self):
        for target_coordinate in self.coordinates:
            for coordinate in self.coordinates:
                if(math.dist(target_coordinate, coordinate) <= self.min_radius):
                    self.direct_density_reachable_count += 1

            if(self.direct_density_reachable_count >= self.min_pts):
                self.core_points_count += 1
                # print("Found core point ie. ", target_coordinate)

            self.direct_density_reachable_count = 0

        if(self.core_points_count >= self.core_pt_threshold):
            self.alert_system.initiateMediumLevelAlert("Found clusters from DBSCAN as core points found is {} and threshold is {}".format(self.core_points_count, self.core_pt_threshold))
            # text_to_speech("Clusters detected")
            self.core_points_count = 0 # Reinitialize

            return True
        else:
            # print(self.core_points_count, " core points found which is less than ", self.core_pt_threshold)
            # print("Coordinates list: ",self.coordinates, " radius = ", self.min_radius, " min pts = ", self.min_pts)
            self.core_points_count = 0 # Reinitialize
            return False
        
    def autoAdjustParameters(self, sample_radius: float):
        """Provide sample radius value for initialization, and coordinates"""
        if(self.single_time_runner%20 == 0): # Update parameters in every 20 times
            self.single_time_runner += 1
            return False
        
        direct_density_reachable_count_array = np.array([])
        direct_density_reachable_count = 0

        for target_coordinate in self.coordinates:
            for coordinate in self.coordinates:
                if(math.dist(target_coordinate, coordinate) <= sample_radius):
                    direct_density_reachable_count += 1

            direct_density_reachable_count_array = np.append(direct_density_reachable_count_array, direct_density_reachable_count)
            direct_density_reachable_count = 0

        self.min_pts = np.mean(direct_density_reachable_count_array)*1.5 # Auto set the min _pts
        self.core_pt_threshold = len(self.coordinates)/2
        if(self.core_points_count >= 30):
            self.core_pt_threshold = 30
        elif(self.core_points_count <= 10):
            self.core_pt_threshold = 12

        self.single_time_runner += 1

engine = pyttsx3.init()
databaseConnector = DatabaseConnector()
databaseConnector.connect()

# Getting video inputs
# video_path = input("Enter the path of video to analyze: ")
# video_path = "Source/ML/accidents/cyberabad_traffic_incident2.mp4"
video_path = "Source/ML/accidents/pakistan_accident_1.mp4"

# video_path = "Source/ML/Crimes/chainSnatch1.mp4"
name = video_path.split("/")[-1].split(".")[0]

databaseConnector.generalInfoTable = "gen_" + name
databaseConnector.specificInfoTable = "sp_" + name

databaseConnector.clearTables()

video_capture = cv2.VideoCapture(video_path)
databaseConnector.create_tables()

# Getting the video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "output/"+video_path.replace(".mp4", "_output.mp4")
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
        self.speed = None

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
                    # objects_detected: dict with class_id as keys, and values being array of objects belonging to that class
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
    return [persons_count, vehicle_count]

def calculate_speed(objects_detected):
    """This function needs to run twice in order to determine the average speed of the vehicle. Make sure to reset object.time_bbox_updates to {} after running this function twice."""
    previous_avg_speeds = []
    objects_detected_ids = list(objects_detected.keys())
    for id in objects_detected_ids:
        obj_of_id = objects_detected[id]
        
        for object in obj_of_id:
            
            obj_len = len(object.time_bbox_updates) # Putting the obj_len'th observation into object.time_bbox_updates
            
            if (obj_len < 2): # This makes it necessary to re-initialize time_bbox_updates to reduce it's length to 0 and make this run again every time it gets to it's max length ie 2
                object.time_bbox_updates[obj_len] = [object.bbox, time.time()]

            else:
                # Inserting real time coordinates of vehicles, putting it inside calculate_speed because calculate_speed runs very fast, which enables us to get a lot of data in just a few milliseconds. If you want to slow down the speed of data collection, jusst move this to the else block, where it'll be executed only once in 0.5 sec
                obj_dist = math.dist(object.time_bbox_updates[0][0][:2], object.time_bbox_updates[1][0][:2]) # Calculating distance between top left coordinates of same object, at different time intervals
                obj_time = object.time_bbox_updates[1][1] - object.time_bbox_updates[0][1]
                try:
                    speed = obj_dist/obj_time
                    previous_avg_speeds.append(speed)
                except ZeroDivisionError:
                    print("Encountered zero division error in line 401 as obj_time is {}".format(obj_time))
                    speed = "NULL"
                object.speed = speed # This will enable other processes to access the speed of object. I I create anymore processes, then MySQL won't work for them (As per the observations), it keeps retrieving same data time and again, in that case, this multiprocessor shared variable can help
                # print("Adding {} to the database".format(object.id))
                databaseConnector.updateSpTable(time.ctime(), object.id, *object.bbox, speed)
    
                    
        objects_detected[id] = obj_of_id
    return previous_avg_speeds # the value returned when this function is executed twice is the final value

def reinitialize_time_bbox_updates(objects_detected):
    """Re-initializes the time bbox updates in order to prevent un-useful data accumulation while executing real time general data collector. Execute this after calculate_speed is run twice. I re-initializes after time_bbox_updates gets 2 elements before if this array doesn't gets cleared, then calculate_speed function won't function."""
    objects_detected_keys = list(objects_detected.keys())
    for id in objects_detected_keys:
        objects_of_id = objects_detected[id]
        for object in objects_of_id:
            object.time_bbox_updates = {}
        objects_detected[id] = objects_of_id

def realTimeGeneralDataCollector(objects_detected, video_processor_active):
    """Records the average speed of vehicles present in the time duration of 2 seconds, along with the count of vehicles and people."""

    # Getting all the data
    start_time = time.ctime()

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
    reinitialize_time_bbox_updates(objects_detected) # To prevent accumulation of data and enable expected functioning of calculate_speed, see calculate_speed() to know more about it's exact functioning and requirement
    end_time = time.ctime()
    print("Attempting database entry...")
    databaseConnector.updateGenTable(start_time, end_time, objects_count[0], objects_count[1], avg_speed)
    print("Data saved")

    if (video_processor_active.value): # If the video is also getting processed simultaneously, terminate the recursion
        realTimeGeneralDataCollector(objects_detected, video_processor_active)

def text_to_speech(text: str):

    engine.say(text)
    engine.runAndWait()

class Alerts:
    def __init__(self):
        self.alertHistory = {} # Keys: time, values: further details of the alert like cause, severity level, alert cause
        self.highLevelAlert = "bold red"
        self.mediumLevelAlert = "bold blue"
        self.lowLevelAlert = "bold yellow"

    def initiateLowLevelAlert(self, cause: str):
        console.print(cause, style = self.lowLevelAlert)

    def initiateMediumLevelAlert(self, cause: str):
        console.print(cause, style = self.mediumLevelAlert)

    def initiateHighLevelAlert(self, cause: str):
        console.print(cause, style = self.highLevelAlert)

def realTimeDetection(objects_detected, video_processor_active, anomalousSpeedMeans: AnomalousMeans, anomalousCountMeans: AnomalousMeans, outlierSpeedDetector: OutlierDetector, outlierCoordinateDetector: OutlierDetector, anomalousStayTimeDetector: AnomalousMeans, dbscan: DBSCAN):
    time.sleep(0.2)
    time.sleep(0.8)
    object_keys = list(objects_detected.keys())
    all_coordinates = []
    # Counting all objects visible on screen
    all_objects_count = 0
    mean_x_width = np.array([]) # To give a sample rad to 
    for id in object_keys:
        all_objects_count += len(objects_detected[id])
        # Iterating through each and every object
        for object in objects_detected[id]:
            all_coordinates.append([object.bbox[0], object.bbox[1]])
            mean_x_width = np.append(mean_x_width, object.bbox[2] - object.bbox[0])
            print("Speed of object: ",object.id , " is ", object.speed)
            anomalousSpeedMeans.autoTrainerChecker(np.array([object.speed]))
            if(outlierSpeedDetector.autoOutlierUpdaterAnomalyChecker(np.array([object.speed]))):
                print("Outlier in speeds")
            if(outlierCoordinateDetector.autoOutlierUpdaterAnomalyChecker(np.array([object.bbox[0], object.bbox[1]]))):
                 # Pass inputs as np array
                print("Outlier in coordinates")
    print("Performing DBSCAN analysis")
    dbscan.coordinates = all_coordinates
    dbscan.autoAdjustParameters(np.mean(mean_x_width))
    dbscan.findClusters()
    print("Ending DBSCAN analysis")

    anomalousCountMeans.autoTrainerChecker(all_objects_count)

    if(video_processor_active.value):
        realTimeDetection(objects_detected, video_processor_active, anomalousSpeedMeans, anomalousCountMeans, outlierSpeedDetector, outlierCoordinateDetector, anomalousStayTimeDetector, dbscan)
    else:
        print("Terminating real time detector")

if __name__ == "__main__":

    # Creating multiple processes
    # Performing a lot of machine learning computations can be too expensive, hence it's better to split it up among multiple processors to ensure a distributed, faster work

    # Creating shared variables among multi processors because even though a variable is made global, it doesn't update for all processor tasks
    manager = multiprocessing.Manager()
    
    objects_detected = manager.dict()
    video_processor_active = multiprocessing.Value("b", True)
    
    # Turns on the camera and starts YOLOv7 detection
    live_yolo_detection_process = multiprocessing.Process(target = start_ai_cam, args = (objects_detected, video_processor_active))
    
    alertSystem = Alerts()
    # Calculates and stores avg speed, crowd, and vehicle count simultaneously
    realTimeGenDataCollector = multiprocessing.Process(target = realTimeGeneralDataCollector, args = (objects_detected, video_processor_active))
    
    # For performing detections, using only objects_detected shared variable as mySQL connection for further processes won't work
    anomalousSpeedMeans = AnomalousMeans(alertSystem) # For checking anomalus changes in the speeds of vehicles. This function compares the mean values of all data points captured, so it should run relatively less number of times
    anomalousSpeedMeans.detectionField = "Speed"
    # anomalousSpeedMeans.anomalyCheckTime = 5 # To keep comparing the means after every 10 sec
    anomalousCountMeans = AnomalousMeans(alertSystem) # For checking anomalous changes in the count of vehicles. This function compares the mean values of all data points captured, so it should run relatively less number of times
    anomalousCountMeans.detectionField = "Vehicle count"
    # anomalousCountMeans.anomalyCheckTime = 5
    outlierSpeedDetector = OutlierDetector() # Can run in real time after inserting proper data to check real time speed anomalies
    outlierSpeedDetector.detectionField = "Speed"
    outlierCoordinateDetector = OutlierDetector() # For detecting objects at anomalous coordinates
    outlierCoordinateDetector.detectionField = "Coordinates"
    outlierCoordinateDetector.anomalyThreshold = 3
    anomalousStayTimeDetector = AnomalousMeans(alertSystem) # Detect when objects stay for just too long under camera survelliance areas. During accidents, objects stop moving, people help the victims, and stay under camera for just too ling. It's made anomalous mean detector because we wanna detect anomaly of staying time for all objects combined, not just 1 or 2 individually.
    dbscan = DBSCAN(100, 8, alertSystem)
    dbscan.core_pt_threshold = 25

    outlierSpeedDetector.alertsInstance = alertSystem
    outlierCoordinateDetector.alertsInstance = alertSystem
    anomalousStayTimeDetector.alertsInstance = alertSystem
    anomalousSpeedMeans.alertsInstance = alertSystem

    realTimeDetec = multiprocessing.Process(target = realTimeDetection, args = (objects_detected, video_processor_active, anomalousSpeedMeans, anomalousCountMeans, outlierSpeedDetector, outlierCoordinateDetector, anomalousStayTimeDetector, dbscan))
    
    # Starting both the processes at the same time

    live_yolo_detection_process.start()
    realTimeGenDataCollector.start()
    realTimeDetec.start()

    # Waiting until both the processes gets finished
    live_yolo_detection_process.join()
    realTimeGenDataCollector.join()

    databaseConnector.disconnect()

    print("Clean exited all tasks being executed by multiple processors")