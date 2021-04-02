# Facial-Expression-Detection

## INTRODUCTION

This project aims to classify six common human facial expression: Happy, Fear, Sad, Angry, Disgust and Surprise, using CNN, a deep convolutional neuron network. The model was trained on our manually collected dataset which is consisted of approximately 60k images. They are portrait, or id (head and face towards the camera) pictures of about 80 volunteers (male and female adults). Each set of the emotion pictures is taken in 3 different light conditions: strong, weak and dark.

The following pipeline was employed to develop the model

- Face extraction from the picture - using dlib library and five point facial landmark;
- Transfer learning using VGG16/ResNet on ImageNet
  +  Feature extraction with convolution base layers
  +  Emotion detection - multi-class emotion classifier
- Fine-tune the new model 
    

This app was developed using Python 3 and deployed using Streamlit. In addition, we use other libraries including Cmake, OpenCV, Tensorflow, Streamlit and so on. For more information on the required packages, you can check requirements.txt

    
## BASIC USAGE

First of all, you can clone our repo by running 
```
git clone https://github.com/vuchau8592/Facial-Expression-Detection.git
```

After cloning the repo, and moving to the Facial-Expression-Detection, you can run our app with 02 options:

### Install required dependencies and run the app on your local computer

With this option, you need to install all the required dependencies in your local computer. To install all the packages, you can run: 
```
pip install -r requirements.txt
```
Then, on Anaconda Prompt, you can run:
```
streamlit run app.py
```
The local host port 8501 will automatically be opened and you can try the app there. If it is not automatic, you can copy this URL and run it http://localhost:8501/.

### Use dockerfile

Generally, docker is used to create a container in which all your codes can  be run without installing any dependencies on your local computer. After installing docker, you can run the following code. You can find more information on docker installation [here](https://docs.docker.com/desktop/). 
```
- docker build . -t [NAME OF THE IMAGE] 
- docker run -p 8501:8501 [NAME OF THE IMAGE]
```
(NAME OF THE IMAGE can be anything you want)
