import numpy as np
import cv2
import sys, getopt
import pafy
import os.path
from os import path
import threading
import time
import datetime

class Detected_Object():
    def __init__(self, boxes, confidence, class_name):
        self.bbox = boxes
        self.confidence = confidence
        self.class_name = class_name
        
        return
    
class Detect_Config():
    ##################### DETECTION CONFIG ##################################################
    def __init__(self):
        
        self.CONF_THRESH = 0.5
        self.NMS_THRESH = 0.5

        # Load the network
        self.net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
        self.classes = []
        with open('coco.names', "r") as f:
            self.classes = f.read().splitlines()
        
        return

def Detect(config, frame):

    # Get variables from configuration class
    CONF_THRESH = config.CONF_THRESH
    NMS_THRESH = config.NMS_THRESH
    net = config.net
    classes = config.classes
 
    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if (confidence > CONF_THRESH):
                
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append((float(confidence)))
                class_ids.append(int(class_id))
    
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    
    if len(indices)>0:
        detected_objects = []
        for index in indices.flatten():
            detected_object = None
            detected_object = Detected_Object(boxes[index],confidences[index],classes[class_ids[index]])
            detected_objects.append(detected_object)
            
            print(detected_object.bbox)
            print(detected_object.confidence)
            print(detected_object.class_name)
            
        return(detected_objects)
    else:
        return

def Take_a_Picture(image_location, original_frame):
    frame = original_frame
    
    # using now() to get current time  
    current_time = datetime.datetime.now()

    for x in range(1):
        image_name = image_location+str(current_time)+".jpg"
        cv2.imwrite(image_name, frame)
    return(image_name)

def Highlight_Reel(image_location):
    
    directories = os.listdir(image_location)
    
    out = cv2.VideoWriter('highlight.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))

    # This would print all the files and directories
    for file in directories:
        location = (str(image_location)+str(file))
        absolute = os.path.abspath(location)
        print(absolute)
        ok, frame = cv2.imread(absolute)
        out.write(frame)
    
    return("highlight.avi")
    
def Config_Stream(stream_URL):
    #Pafy Config
    vPafy = pafy.new(stream_URL)
    play = vPafy.getbest()
    
    # Read video
    video = cv2.VideoCapture(play.url)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    return(video)

def Config_File(input_file):
   
    # Read video
    video = cv2.VideoCapture(input_file)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    return(video)    

def Helper(config, frame, image_location):
    detected_objects = []
    detected_objects = Detect(config, frame)

    classes = config.classes
    if detected_objects:
        annotation=True
        if annotation:
            colors = []
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            
            for i, detected_object in enumerate(detected_objects,1):
                x, y, w, h = detected_object.bbox
                print (x,y,w,h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2)
                cv2.putText(frame, str(round(detected_object.confidence, 2)) , (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[i], 2)
                
        
        print("Taking a picture...")
        image_name = Take_a_Picture(image_location, frame)
        print(image_name)

def main(argv):
    
    #Read arguments
    input_file = ''
    stream_URL = ''
    image_location = "./images/"
    
    try:
        opts, args = getopt.getopt(argv,"hs:f:",["stream_URL=","file="])
    except getopt.GetoptError:
        print ('test.py -s <stream_URL> OR -f <input_file_path>')
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -s <stream_URL> OR -f <input_file_path>')
            sys.exit()
        elif opt in ("-s", "--stream_URL"):
            stream_URL = str(arg)
            print ('Stream URL is ', stream_URL)
        elif opt in ("-f", "--file"):
            input_file = str(arg)

    
    
    print ("Is "+image_location+" a Directory?")
    if path.isdir(image_location):
        print("Yes")
    else:
        print("No, making directory.")
        os.mkdir(image_location)


    if stream_URL:
        video = Config_Stream(stream_URL)
    elif (input_file):
        video = Config_File(input_file)

    
    
    # Configure Detection
    config = Detect_Config()

    # Detect object and Snap
    detect_counter = 61
    NMSBoxes = None

    cv2.namedWindow("Snap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snap", 1280, 720)

    while True:
        ok, frame = video.read()
        
        NMSBoxes = None

        if (detect_counter > 60):
            x = threading.Thread(target=Helper, args=(config, frame, image_location))
            print("starting a Helper thread")
            x.start()
            x.join()
            detect_counter = 0
            
        detect_counter = detect_counter + 1
        

        """ if (photo_id > 2):
            print("Building highlight reel...")
            video_name = Highlight_Reel(image_location)
            if video_name:
                print("Done!")
                print(video_name)
            break
        """


        time.sleep(int(1/30))
        cv2.imshow("Snap", frame)
        
        k = cv2.waitKey(1) & 0xff
        if (k == 27) : 
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
   main(sys.argv[1:])