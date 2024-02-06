
# Project Overview

This system can track the movement and behavior of vehicles in different environments. The system typically uses cameras to capture visual information, and then the object detection model analyzes this data to track and monitor the vehicles in real time or post-processing.


## Vehicle Detection

• Used pre-trained model YOLOv5 to detect vehicles because, it's primarily designed for object detection tasks, where it can detect and locate multiple objects in an image or video frame simultaneously offering a good balance between accuracy and speed.

• Used the YOLOv5s variant to manage the user to apply the system on the CPU. It has fewer parameters and requires less computational power compared to larger variants.

## Vehicles Tracking
- Created a tracker class to assign unique IDs for each car and update the bounding boxes to track the cars accurately.
- It assigns a unique ID for each car by calculating the center points of each car and comparing the center points to every car detected to check if it has been detected before or not using the Euclidean norm.
- After processing all the bounding boxes, the class removes all the IDs that are not associated with any objects in the current frame.


