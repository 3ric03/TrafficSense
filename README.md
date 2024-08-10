## Overview
This project features a two-stage deep learning system designed to detect and classify 26 common traffic sign types from streetview photos across the US and Canada. The model is developed using PyTorch and trained on Google Colaboratory. The training data is sourced from a variety of open-source datasets, including those from Roboflow and Mapillary. For a technical breakdown of the project and a detailed performance analysis, please refer to the report available [here](https://drive.google.com/file/d/10MgrnIAf6UcEKIky3fRQqqh63al1jnA5/view?usp=sharing).

## Architecture
### Stage One: Locating a traffic sign within an image using YoloV8
We used a pre-trained YOLOv8 model from Ultralytics to locate traffic signs in images. YOLOv8, originally trained on the COCO dataset with 80 object classes, was fine-tuned with a Roboflow dataset containing 26 traffic sign classes. This allows the model to output precise bounding box coordinates for traffic signs within an image.
### Stage Two: Feed the cropped image to our own Convolutional Neural Network model
We developed and trained a custom Convolutional Neural Network (CNN) to classified the cropped image from stage one. The encoder features four convolutional layers, each followed by a 2x2 max pooling layer with a stride of 2. This setup produces a 16x1x1 feature map, capturing high-level features. This feature map is then fed into a small ANN classifier with 244 neurons and produces an output layer of 26 classes.

<img src="https://github.com/3ric03/TrafficSense/blob/main/img/model_architecture.jpg" width="1036px" height="500px"> 

## Input and Output Example
In this simple example, we passed in a street view image with a Canadian stop sign, our program is able to bound the sign and classify it correctly with 99% confidence.

<img src="https://github.com/3ric03/TrafficSense/blob/main/img/q_normal_cond.png"> 

## Data Processing and Model Training
To ensure consistency, all training images (approximately 18,000) were resized to 150x150 pixels. We used an 80-10-10 split for training, validation, and testing.
Here are the training & validation loss and error graphs.

<img src="https://github.com/3ric03/TrafficSense/blob/main/img/error.png" width="500px" height="400px"> <img src="https://github.com/3ric03/TrafficSense/blob/main/img/loss.png" width="500px" height="400px">
