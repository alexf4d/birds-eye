# birds-eye

## Table of Contents
- [Introduction](Introduction)
- [Dependencies](Dependencies)

---

## Introduction
A computer vision project to analyze birds.

A software suite of the following:

1. Snap:
     - Scrub through live and recorded video, detect birds, and then save pictures.

2. Journal:
    1. Detect birds in live video creating a log of birds seen date/time stamped
    2. Use the log to build graphs and highlight trends of common feeding times by time of day, day of week, month, season, and year.
    
    - For the future:

      1. Train a new AI model to recognize the type of bird adding more granularity to the logs, graphs, and trends

3. Notify:
    1. Provisions for ITTT integration to notify when a bird is at the feeder acompanied by a photo.
       - (Could be user expanded to things like giving your birds a Twitter account.)


---

## Dependencies
openCV - https://github.com/opencv/opencv-python

- `$ sudo apt-get install python3-openCV`

Darknet - https://github.com/AlexeyAB/darknet

- Follow build instructions at the repository

### Pre-trained Models

Some pretrained models are required. These should be added to the root of the project.

- [yolov3.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg)

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

- [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)

