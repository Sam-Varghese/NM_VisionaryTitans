# YOLOv7 surveillance system, with AI cameras, powered by YOLOv7
# Leverages multiple processors

# OpenCV Live feed version

import cv2
import time
import random
import numpy as np
import onnxruntime as ort
# import multiprocessing
import tensorflow as tf
import os
import pyttsx3

engine = pyttsx3.init()

print("Imported necessary libraries...")

cuda = True # For utilizing GPU, and performing parallel computing
weights = "./yolov7-tiny.onnx"
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
# classes = ["person"]
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

# Calculating time taken to process each frame
start_time = time.time()
img_counter = 1

# For OpenCV fonts
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.0

# OpenCV camera capture
def start_ai_cam():
    try:

        # Starting OpenCV Video Capture
        print("Initiating camera...")
        capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            print("Camera being used by another application, unable to gain access")
            exit()

        screen_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        screen_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Create a named window for full screen display
        cv2.namedWindow("Live Footage", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Live Footage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Setting frame as a global variable to make it accessible to multiple processors 
        global frame
        global detected_frames_list
        detected_frames_list = []
        img_counter = 1

        while True:

            ret, frame = capture.read()
            # frame_cpy = frame.copy()
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

            for i, prediction_array in enu_predic:
            
                for (batch_id, x0, y0, x1, y1, cls_id, score) in prediction_array:
                    # Coordinates are of top left and bottom right

                    if score < 0.5:
                        continue

                    class_name = all_classes[int(cls_id)]

                    if class_name in classes:

                        # Saving the image
                        # roi = frame[int(y0): int(y1), int(x0): int(x1)]
                        # np.concatenate(person_frames_list, roi)
                        # person_frames_list.append(roi)
                        # cv2.imwrite("./people/Sam/{}_image.jpg".format(img_counter), roi)

                        class_color = colors[class_name]

                        # Reversing the paddings and other transformations applied during letterbox

                        box = np.array([x0,y0,x1,y1])
                        box -= np.array(dwdh*2)
                        box /= ratio
                        box = box.round().astype(np.int32).tolist()

                        roi = frame[box[1]: box[3], box[0]: box[2]]
                        detected_frames_list.append(roi)
                        # cv2.imwrite("./people/Sam/{}_image.jpg".format(img_counter), roi)

                        cv2.rectangle(frame, box[:2], box[2:], class_color, thickness)
                        cv2.putText(frame, class_name, box[:2], font, font_scale, class_color, thickness)

            cv2.imshow("Live Footage", frame)

            img_counter += 1
            if (cv2.waitKey(1) == ord("q")):
                end_time = time.time()
                print("Avg frame processing time (time taken/ frames processed): ",(end_time- start_time)/img_counter)
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

def get_folder_names(path: str):

    folder_names = []

    for entry in os.scandir(path):

        if(entry.is_dir()):
            folder_names.append(entry.name)

    return folder_names

def train_ml_model():

    print("Initiating model training")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Pass normalized image here

    classes_names = get_folder_names("./people")

    # Creating image data generators

    datagen = ImageDataGenerator(
        rescale = 1./255, # Normalizing pixel values
        rotation_range = 10, 
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.2,
        horizontal_flip = True
    )

    # Flowing images from directory using ImageDataGenerator

    train_generator = datagen.flow_from_directory(
        "./people",
        target_size = (244, 244),
        batch_size = 16,
        class_mode = "sparse"
    )

    ml_model = tf.keras.Sequential([
        # Adding neural network layers

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(len(classes_names), activation = "softmax")
    ])

    ml_model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
        metrics = ["accuracy"]
    )

    ml_model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples, # returns the count of total number of images in the train dataset
        epochs = 30
    )

    ml_model.save("./savedModels/model.h5")

def text_to_speech(text: str):

    engine.say(text)
    engine.runAndWait()

def roi_tasks(save_images = False):
    """After getting ROI (Region of Interest) image of the user, perform image identification and further proceedings. It will be accessing global variables from start_ai_cam(), to perform further computations."""

    try:
        ml_model = tf.keras.models.load_model("./savedModels/model.h5")

    except Exception as e:
        print("Encountered exception while accessing save ml model. Creating new ML model")
        train_ml_model()

    finally:
        ml_model = tf.keras.models.load_model("./savedModels/model.h5")

    # Predicting faces...
    predictions = np.array()

    # Normalizing entire person detection frames
    person_frames_list /= 255

    for face in person_frames_list:

        prediction = ml_model.predict(face)
        predictions.append(prediction)

    print("All predictions: ", predictions)

if __name__ == "__main__":

    start_ai_cam()
    # print("Starting")
    # text_to_speech("Sam")
    # train_ml_model()