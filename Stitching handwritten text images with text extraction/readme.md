
# Project Overview

Creating panoramic images by stitching handwritten text
images using the SIFT algorithm with Homography matrix approach as well as applying an OCR model for extracting text from the panoramic image.


## Image Stitching

Image stitching is a computer vision and image processing technique that involves combining multiple images with overlapping fields of view to create a panoramic or wide-angle image. The goal of image stitching is to seamlessly merge these individual images into a cohesive and visually consistent panorama.

Multiple images are captured by a camera or other imaging device, often with some degree of overlap between consecutive images.
Feature Extraction such as corners or distinctive points, are identified in each image. These features serve as reference points for aligning and merging the images.

![mk](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme1.png)

The images are then aligned based on the extracted features. This involves finding the transformation (translation, rotation, scaling) that minimizes the differences between corresponding features in different images.

After alignment, the overlapping regions of the images need to be blended together to create a seamless transition. This can involve techniques such as feathering, gradient blending, or more advanced algorithms that consider color and intensity variations.

Additional corrections may be applied to address distortions, vignetting, or other artifacts that can occur during image capture.

The final result is a panoramic image that provides a wide-angle view, often covering a larger field of view than any single image in the original set.

![mN](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme3.png)

### SIFT Algorithm
The Scale-Invariant Feature Transform (SIFT) algorithm is a computer vision algorithm used for image feature detection, description, and matching.the detector can still distinguish the corners even if the image is rotated. However, the Harris Detector cannot perform well if the image is scaled differently.

![SI](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme2.png)

SIFT starts by building a scale-space representation of an image using Gaussian blurring at different scales. This helps in detecting features at different sizes.
The Difference of Gaussians (DoG) is computed by taking the difference between consecutive blurred images in the scale-space pyramid. When the DoG is found, the SIFT detector searches the DoG over scale and space for local extremas, which can be potential key points. Local extrema in the DoG pyramid are identified as potential key locations.
```bash

descriptor = cv2.SIFT_create()
(keypoints, features) = descriptor.detectAndCompute(image, None)

```
### What is the difference between key points and descriptors?
One important thing to understand is that after extracting the key points, you only obtain information about their position, and sometimes their coverage area (usually approximated by a circle or ellipse) in the image. While the information about key points ' position might sometimes be useful, it does not say much about the key points themselves.

Depending on the algorithm used to extract key points (SIFT, Harris corners, MSER), you will know some general characteristics of the extracted key points (e.g. they are centered around blobs, edges, prominent corners...) but you will not know how different or similar one key point is to the other.

So, here come descriptors: they are the way to compare the key points. They summarize, in vector format (of constant length) some characteristics of the keypoints. For example, it could be their intensity in the direction of their most pronounced orientation. It assigns a numerical description to the area of the image the keypoint refers to.

#### They should be independent of key points ' position
If the same key point is extracted at different positions (e.g. because of translation) the descriptor should be the same.

#### They should be robust against image transformations
Some examples are changes of contrast (e.g. image of the same place during a sunny and cloudy day) and changes of perspective (image of a building from center-right and center-left, we would still like to recognize it as the same building).

Of course, no descriptor is completely robust against all transformations (nor against any single one if it is strong, e.g. big change in perspective).

Different descriptors are designed to be robust against different transformations which is sometimes opposed to the speed it takes to calculate them.

### Brute force matcher
BF Matcher matches the descriptor of a feature from one image with all other features of another image and returns the match based on some distance calculation. So in another word, given 2 sets of features (from image 1 and image 2), each feature from set 1 is compared against all features from set 2. It is slow since it checks match with all the features.

It is a simple technique to decide which feature in the query image is best matched with that in the train image. This perfect match is elected by looking at the smallest distance among those computed among one feature in the train pic and all the features in the query pic.

Brute-Force matcher is simple. It takes the descriptor of one feature in the first set and is matched with all other features in the second set using some distance calculation. And the closest one is returned.

For BF matcher, first, we have to create the BFMatcher object using cv2.BFMatcher(). It takes two optional parameters.

The first one is normType. It specifies the distance measurement to be used. By default, it is cv2.NORM_L2. It is good for SIFT, SURF etc (cv2.NORM_L1 is also there).

For binary string-based descriptors like ORB (Oriented FAST and Rotated BRIEF), BRIEF(Binary Robust Independent Elementary Features), BRISK, etc, cv2.NORM_HAMMING should be used, which uses Hamming distance as measurement. If ORB is using VTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used.
```bash

 bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
 best_matches = bf.match(features_first_img,features_second_img)

```
Two important approaches BFMatcher.match() and BFMatcher.knnMatch(), the first returns the best match, and the second method returns k best matches.
For cases where we want to consider more than one candidate match, we can use a KNN-based matching procedure. Instead of returning the single best match for a given feature, KNN returns the k best matches. Note that the value of k has to be pre-defined by the user. As we expect, KNN provides a larger set of candidate features. However, we need to ensure that all these matching pairs are robust before going further.

Which means, instead of a list of matches you get a list of a list of matches.

### Homography Matrix
Homography is a transformation that maps the points in one image to the corresponding point in another image. The homography is a 3×3 matrix :

![HM](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme4.png)

Consider two images of a plane (top of the book) shown in Figure 1. The red dot represents the same physical point in the two images. In computer vision jargon we call these corresponding points. Figure 1. shows four corresponding points in four different colors — red, green, yellow, and orange. A Homography is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.

![HH](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme5.png)

Homography matrix simply represents the transformation of one point in an image plane to the same point in another image plane.


To calculate the homography between two images, we must know at least four corresponding points between them. OpenCV robustly estimates a homography that fits all corresponding points in the best possible way. The point correspondences are found by matching features like SIFT or SURF between the images.

![HK](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme6.png)

![HL](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme7.png)

### RANSAC algorithm
RANdom SAmple Consensus or RANSAC is an iterative algorithm to fit linear models. Different from other linear regressors, RANSAC is designed to be robust to outliers. Here, I will use RANSAC to estimate the Homography matrix. Note that Homography is very sensitive to the quality of data we pass to it, hence need an algorithm (RANSAC) that can filter irrelevant points from the data distribution.

### RANSAC for Homography
Repeat N times :
1- select 4 random pairs of matched features.
2- Fit Homography.
3- Compute inliers: apply the transformation to all features and calculate the error (distance) between matching points after the transformation.
4- Count the number of inliers then select the homography matrix with the largest number of inliers.

```bash
    if len(matches) > 4:
        # construct the two sets of points
        points_first = np.float32([keypoints_first_img[m.queryIdx] for m in matches])
        points_second = np.float32([keypoints_second_img[m.trainIdx] for m in matches])

        # Calculate the homography between the sets of points
        (H, status) = cv2.findHomography(points_first, points_second, cv2.RANSAC, reprojThresh)
        
```

## Image warping
After the Homography Matrix calculation, need to warp one of the images to a common plane
Finally, we can apply our transformation by calling the cv2.warpPerspective function. The first parameter is our original image that we want to warp, the second is our transformation matrix M (which will be obtained from homography_stitching), and the final parameter is a tuple, used to indicate the width and height of the output image.

![IW](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/readme8.png)

After calculating the transformation matrix (which in this case is the Homography_Matrix), apply the perspective transformation to the entire input image to get the final transformed image.

```bash
result = cv2.warpPerspective(first_photo, Homography_Matrix,  (width, height))
```

### Image blending
Blending the two images by directly replacing the corresponding region in the transformed first photo with the second photo.

```bash
result[0:second_photo.shape[0], 0:second_photo.shape[1]] = second_photo
```

## Results of image stitching 
#### Input Images : 

![e1](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/stif_ex2.png)

![e2](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/stif_ex35.png)

#### Output panoramic images :

![r1](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/stif_res2.png)

![r2](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/stif_res3.png)


## Applying an OCR model 
### 1 - Pytesseract 
First, I tried  using pytesseract which is a widely used open-source OCR (Optical Character Recognition) engine.

```bash
image_path = 'panorama_img_.jpeg'
img = Image.open(image_path)

# Perform OCR
text = pytesseract.image_to_string(img)
``` 
But it was a bad option for extracting text from handwritten text images, and this is an example :

![r2](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/p1.jpeg) 

### 2- EasyOCR

EasyOCR is built on top of PyTorch and leverages deep learning techniques for text recognition. EasyOCR comes with pre-trained models for different languages and scripts, making it easy to get started without the need for extensive training.

```bash
reader = easyocr.Reader(['en'], gpu = True)
img = cv2.imread('panorama_img_.jpeg')
results = reader.readtext(img)
for result in results:
    # Extract text and bounding box
    text = result[1]
    if len(text) > 1:
        print (text)
    box = result[0][-4:]
    box = [(int(x), int(y)) for x, y in box]

    # Draw bounding box
    img = cv2.polylines(img, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw text with an offset below the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (10, 20, 0)
    text_position = (box[0][0], box[0][1])  # Adjust the offset (e.g., 20 pixels)
    img = cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Display the image with bounding boxes and text
plt.imshow( img)
```
![EaL](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/OCR_res7.png)

### 3- KerasOCR 
Used KerasOCR and it brought the best accuracy for this task but it needs GPU acceleration so, I used Google Colab.
KerasOCR is a Python package that provides a high-level API for training and using a text detection and recognition pipeline. It is based on the Keras framework and uses the CRAFT text detector and the CRNN recognition model. 

```bash
image = keras_ocr.tools.read('/content/panorama_img_.jpeg')
fig, ax = plt.subplots(1, 1, figsize=(15, 30))
# Create an OCR recognizer
pipeline = keras_ocr.pipeline.Pipeline()
# Perform OCR on the image
prediction_groups = pipeline.recognize([image])
# Draw bounding boxes on the image
keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)
ax.set_title('kerasocr results', fontsize=24)

plt.imshow(image)
plt.axis('off')
plt.show()
```
![EaL](https://github.com/Abdelrahmann94/Computer-vision-/blob/main/kk.png)

## Conclusion 
I used SIFT Algorithm with the Homography matrix technique to apply seamless image stitching for handwritten text images. Then applied an OCR model using KerasOCR.

## Comments
This approach for image stitching doesn't work properly on images that contain big font sizes, and I put an example of this in the evaluation report.

## Time Allocation

#### Monday :
Received the task and did some research to decide which approach I would use for this task (3 hours)
#### Tuesday : 
Developed an algorithm for image stitching (5 hours)
#### Wednesday :
Tested the image stitching approach and applied an OCR model for the panoramic image (3 hours)
#### Thursday : 
Tested the OCR models and chose the one with the best accuracy,
prepared the evaluation report for the OCR models and image stitching technique (3 hours)
#### Saturday :
prepared a readme file for this task and recorded the video (3 hours)


