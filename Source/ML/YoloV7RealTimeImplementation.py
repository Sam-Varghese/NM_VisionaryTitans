# YOLOv7 surveillance system, with AI cameras, powered by YOLOv7
# Leverages multiple processors

# OpenCV Live feed version

import cv2
import time
import random
import numpy as np
import onnxruntime as ort
# import multiprocessing
import os
import pyttsx3

engine = pyttsx3.init()

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
# classes = ["person", "bus", "bicycle", "traffic light", "car"]
classes = all_classes

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

def distance(bbox1, bbox2):
    return (((bbox2[1] - bbox1[1])**2) + (bbox2[0] - bbox1[0])**2)**0.5

# Creating class to store details of each and every detected object
class Object:
    def __init__(self, object_type, class_id, bbox_coordinates):
        self.id = object_type + str(random.randint(0, 100)) + chr(random.randint(97, 122))
        self.class_id = class_id
        self.bbox = bbox_coordinates

    def find_worthy_child(self, new_bbox_list):
        # print("Self BBOX: ", self.bbox)
        # print("Global bbox list: ", new_bbox_list)
        # input("Press enter to continue: ")
        min_dist = abs(self.bbox[2] - self.bbox[0])/2.5
        selected_element_index = None
        counter = 0
        # print("Starting anew: ")
        for new_bbox in new_bbox_list:
            dist = distance(self.bbox, new_bbox)
            if(dist <= min_dist):
                # Detected worthy child
                min_dist = dist
                worthy_child_bbox = new_bbox
                selected_element_index = counter

            counter += 1
        # If no bbox has been found as a worthy child
        if(selected_element_index == None):
            return None

        # If a bbox is found worthy
        self.bbox = worthy_child_bbox
        return selected_element_index

# For OpenCV fonts
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.0

# Calculating time taken to process each frame
start_time = time.time()
img_counter = 1

# OpenCV camera capture
def start_ai_cam():
    try:

        # Starting OpenCV Video Capture
        print("Initiating camera...")
        capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            print("Camera being used by another application, unable to gain access")
            exit()

        # Create a named window for full screen display
        cv2.namedWindow("Live Footage", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Live Footage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Setting frame as a global variable to make it accessible to multiple processors 
        img_counter = 1

        # Creating a list of instances of object that have been detected
        objects_detected = {}

        while True:

            ret, frame = capture.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # YOLO requires BGR images to be in RGB form

            resize_data = []

            image_cpy, ratio, dwdh = letterbox(image, auto=False)

            image_cpy = image_cpy.transpose((2, 0, 1))
            image_cpy = np.expand_dims(image_cpy, 0) # Adds an extra dimension to image at index 0, YOLOv7 format
            image_cpy = np.ascontiguousarray(image_cpy) # Changes the matrix structure of image_cpy, again YOLOv7 format
            image_cpy = image_cpy.astype(np.float32)

            resize_data.append((image_cpy, ratio, dwdh))

            # Running batch 1 inference

            image = np.ascontiguousarray(resize_data[0][0]/255) # Normalizing the image
            prediction = session.run(outname, {"images": image})

            enu_predic = enumerate(prediction)

            new_bbox = {}
            # colors_cls_id_name = []

            for i, prediction_array in enu_predic:
                
                # If n objects gets detected in the frame, then prediction_array will contain n arrays inside it containing batch_id, x0, y0, x1, y1, cls_id, score

                for (batch_id, x0, y0, x1, y1, cls_id, score) in prediction_array:
                    # Coordinates are of top left and bottom right

                    # Allows only those detections to be shown as output whose confidence value is above a threshold
                    if score < 0.5:
                        continue
                    
                    cls_id = int(cls_id)
                    class_name = all_classes[cls_id]

                    if class_name in classes:

                        # class_color = colors[class_name]

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

                try:
                    # Allowing only the class_id objects to find their worthy child
                    for obj in objects_detected[class_id]:
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

                except Exception:
                    pass

                # Creating new objects for new bboxes discovered that didn't prove as a worthy child of any existing object
                for unused_bbox in new_bbox[class_id]:
                    # Create objects of only those types, which are in the list of classes to get detected
                    if(class_name not in classes):
                        continue

                    obj = Object(class_name, class_id, unused_bbox)

                    try:
                        objects_detected[class_id].append(obj)
                    except Exception:
                        objects_detected[class_id] = [obj]

                    print("Creating new object instance {}".format(obj.id))

                    cv2.rectangle(frame, obj.bbox[:2], obj.bbox[2:], class_color, thickness)
                    cv2.putText(frame, obj.id, obj.bbox[:2], font, font_scale, class_color, thickness)

            cv2.imshow("Live Footage", frame)

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

        capture.release()
        cv2.destroyAllWindows()

def text_to_speech(text: str):

    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":

    start_ai_cam()