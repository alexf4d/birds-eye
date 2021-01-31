import numpy as np
import cv2
import sys, getopt
import pafy
import os.path
from os import path


def Detect(frame):
    ##################### DETECTION CONFIG ##################################################

    CONF_THRESH, NMS_THRESH = 0.5, 0.5

    # Load the network
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    try:
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
        
        NMSBoxes = []
        for index in indices:
            NMSBoxes.append(b_boxes[index])

    except:
        return

    return(NMSBoxes)

def Take_a_Picture(image_id, video, original_frame):
    frame = original_frame
    for x in range(1):
        image_name = image_id+"-"+str(x+1)+".jpg"
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


    # Detect object and Snap
    detect_counter = 31
    NMSBoxes = None
    photo_id = 1

    while True:
        ok, frame = video.read()
        
        NMSBoxes = None
       
        if (detect_counter > 30):          
            NMSBoxes = Detect(frame)
            detect_counter = 0

            if NMSBoxes:
                annotation=False
                if annotation:
                    with open('coco.names', "r") as f:
                        classes = [line.strip() for line in f.readlines()]
                    colors = np.random.uniform(0, 255, size=(len(classes), 3))

                    for index in NMSBoxes:
                        x, y, w, h = NMSBoxes[index]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[index], 2)
                        cv2.putText(frame, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)
                
                print("Taking a picture...")
                image_name = Take_a_Picture(image_location+str(photo_id), video, frame)
                photo_id=photo_id+1
                print(image_name)

                snappy_messages= ["Snapping a Photo!", "Cheese!", "Taking the Picture.", "Shake your tail feathers!"]
                cv2.rectangle(frame, (0, 0), (0 + 1280, 0 + 720), 0, 2)
                cv2.putText(frame, snappy_messages[np.random.randint(0,len(snappy_messages))], (25, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 100, 2)
        
        detect_counter = detect_counter + 1
        

        """ if (photo_id > 2):
            print("Building highlight reel...")
            video_name = Highlight_Reel(image_location)
            if video_name:
                print("Done!")
                print(video_name)
            break
        """


        cv2.namedWindow("Snap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Snap", 1280, 720)
        cv2.imshow("Snap", frame)
        
        k = cv2.waitKey(1) & 0xff
        if (k == 27) : 
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
   main(sys.argv[1:])