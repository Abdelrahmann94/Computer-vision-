
# Project Overview

Egyptian ID documents scanner system that can detect IDs, correct the ID's rotation if it is rotated, segment the text lines in the ID individually, and crop the segmented text lines images.


## Egyptian ID detection

ID Detection using the Contour-based technique. This technique achieved better IoU (intersection over union) than the first approach.

Contours are utilized to identify the boundaries of an ID card within an image. The method involves a series of image-processing steps to isolate and highlight the potential ID card region. Here's a breakdown of the key steps:

- Thresholding and Blurring:
Converted The input image to grayscale, then applied Gaussian blurring. This process helps in reducing noise and simplifying the image for further analysis.

- Contour Detection:
Contours are then detected using the specified threshold. Contours represent the boundaries of distinct regions in the image, Used 'findContours' function to identify these contours.

- Contour Sorting:
Sorted the detected contours based on their areas in descending order. Larger contours are prioritized, as they are more likely to correspond to the ID card in the image.

- Identification of ID Contour:
Implemented a searching algorithm to find the largest closed contour with four sides, indicative of an ID card shape. The technique considers contours with an area greater than a predefined threshold. If the contour meets these criteria, it is identified as the potential contour of the ID card.

- Drawing the Contour:
The identified ID contour is then drawn on the original image using the 'drawContours' function, providing a visual representation of the detected ID card region.

Multiple images are captured by a camera or other imaging device, often with some degree of overlap between consecutive images.
Feature Extraction such as corners or distinctive points, are identified in each image. These features serve as reference points for aligning and merging the images.

## ID' rotation detection and correction


- Used the ' four_point_transform ' Function which performs a perspective transformation, commonly known as a "four-point transform." This transformation rectifies any rotation or skewing present in the ID card. The four corner points of the ID card act as reference points for the transformation.

- The ' four_point_transform' function uses the 'cv2.getPerspectiveTransform' and 'cv2.warpPerspective' functions to perform the perspective transformation.

- It calculates the perspective transform matrix and applies the perspective transformation to correct the rotation of the ID and crop the detected ID from the image. It takes the image and the ID's contour as input parameters.

```bash
ID_Img = four_point_transform(image, ID_contour.reshape(4, 2))
plt.imshow(cv2.cvtColor(ID_Img, cv2.COLOR_BGR2RGB))

```
## Text Segmentation

- Used pre-trained model YOLOv8m-seg for segmenting the text in the ID.
- Trained the model on annotated Egyptian IDs dataset for segmenting the text in IDs
 ![I](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/Egyptian%20ID%20Documents%20Scanner/results.png)


## Cropping the segmented text images

1 - Extracted the data of the predicted segmentation boxes such as the bounding boxes coordinates, confidences, and classes.

2 - Implemented a function to crop all the individual segmented text lines using the bounding boxes coordinates.




