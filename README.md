# birds-eye <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Introduction](#1-introduction)
- [2. Dependencies](#2-dependencies)
  - [2.1 Packages](#21-packages)
  - [2.2 Pre-trained Models](#22-pre-trained-models)
- [3. User Guides](#3-user-guides)
- [4. Educational References](#4-educational-references)

## 1. Introduction
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

## 2. Dependencies
### 2.1 Packages
Python 3 - https://www.python.org/downloads/

numpy - https://github.com/numpy/numpy
   - `sudo apt-get install python-numpy`

Pafy - https://github.com/mps-youtube/pafy

   - `sudo pip install pafy`

openCV - https://github.com/opencv/opencv-python

   - `sudo apt-get install python3-openCV`

Darknet - https://github.com/AlexeyAB/darknet

   - Follow build instructions at the repository
  

### 2.2 Pre-trained Models

Some pretrained models are required. These should be added to the root of the project.

   - [yolov3.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg)

   - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

   - [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)

## 3. User Guides

   - [Snap](https://github.com/alexf4d/birds-eye/blob/main/snap-user-guide.md)
  
## 4. Educational References

Here are some references that were useful in the development process.

   - [Python: Real Time Object Detection (Image Webcam, Video files) with Yolov3 and OpenCV](https://www.youtube.com/watch?v=1LCb1PVqzeY)
