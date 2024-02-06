
# Project Overview

This system can track the movement and behavior of vehicles in different environments. The system typically uses cameras to capture visual information, and then the object detection model analyzes this data to track and monitor the vehicles in real time or post-processing.


## Dataset

- Trained the model on the UCF50 dataset which includes 50 different action categories, covering a variety of human actions and activities.

- For each action category, there are multiple video clips capturing instances of that particular action. The videos vary in terms of duration, resolution, and background.

## Implementing an action recognition model
### CNN-LSTM approach

- Used the CNN-LSTM approach to implement my model. The CNN LSTM architecture involves using Convolutional Neural Network layers for feature extraction on input data combined with LSTMs to support sequence prediction.
- This approach uses the CNN as a features extractor "encoder" and the last hidden layer to be the input to LSTM layer "decoder".

### Modifications enhanced the model's performance


- Resized the frames to be 96x96 instead of 64x64, increasing the input shape helped the model to capture more detailed information and finer patterns in the data. Couldn't increase the input shape more than that due to the limited RAM of Google Colab's sessions.

- Removed the BatchNormalization layer. Found out that it is less suitable for action recognition models. It operates independently on each mini-batch and may disrupt the temporal dependencies in the sequence. This can lead to suboptimal performance in capturing the dynamics of actions over time.

- Added another dense layer with 128 neurons and Relu as an activation function before the last layer to enhance the model's ability to capture complex patterns and representations in the input data and increase the model's capacity.

