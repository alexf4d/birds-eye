# birds-eye
A computer vision project to analyze birds.

Utilizing:
openCV -  https://github.com/opencv/opencv-python
Darknet - https://github.com/AlexeyAB/darknet

Some pretrained models are required. These should be added to the root of the project.
https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
https://pjreddie.com/media/files/yolov3.weights
https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

A software suite of the following:

1. Snap:
    a) Scrub through live and recorded video, detect birds, and then save pictures.

2. Journal:
    a) Detect birds in live video creating a log of birds seen date/time stamped
    b) Use the log to build graphs and highlight trends of common feeding times by time of day, day of week, month, season, and year.
    
    For the future:
    c) Train a new AI model to recognize the type of bird adding more granularity to the logs, graphs, and trends

3. Notify:
    a) Provisions for ITTT integration to notify when a bird is at the feeder acompanied by a photo.
        (Could be user expanded to things like giving your birds a Twitter account.)

