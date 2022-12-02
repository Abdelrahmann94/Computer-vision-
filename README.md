# weapon detector using MobileNetV2-
first ,I gathered guns dataset, knives dataset and some datasets not including weapons then merged them into one and divided the dataset into two classes weapon or no weapon.

if the model detected that there is a weapon , the alarm will turn on .used MobileNetV2 for the detection and it gave me 97.4% accuracy after fine tuning.
used the alarm by importing mixer function from pygame to initiate the sound file.
