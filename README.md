# DeepFake-Detection-Model
Deepfake detection model deployed on Plotly Dash
●	Deepfake refers to a technology that creates fake elements in a video or an image and make it look like a genuine image/video.
●	This code involves using Deep learning Techniques to implement Deepfake detection model to differentiate between fake and real content.
●	The dataset was obtained from Kaggle through a Deepfake Detection Challenge
●	Certain Data preparation techniques have been used to preprocess the data. 
●	Deep learning models only take images as input therefore the video dataset was split into frames.
●	OpenCV library was used to import VideoCapture. This enabled the creation of frames dataset from the videos
●	The video was split and every fourth frame was captured inorder avoid redundancy of data.
●	The face alone was detected and cropped from the frames to remove unnecessary information as DeepFakes mainly involves alterations in facial features. 
●	OpenCV’s HaarCascade classifier was used to detect and crop faces from images. 
●	The data was converted into float values for normalisation.
●	The face data was prepared and was ready to be trained in the model.
●	The model used for this project is EfficientNet b0.
●	The model is highly scalable and produces good results. 
●	The model is trained for 30 epochs with 224x224 image resolution.
●	The train accuracy achieved from the model is 99.1% and validation accuracy was 98.7 %
●	The test accuracy obtained by the model is 61%.
●	The model weights were stored and were deployed in plotly dash for an interactive user interface. 

File info:

●	DeepFake detection on balanced dataset.ipynb - Data preparation and Model training on train and test dataset. 
●	DeepFake B0 function.ipynb - Custom function created for the entire deepfake detection process.
●	Dash_deepfore_frontend.py - Model deployment using plotly dash through interactive dashboard
●	Split_video.py - Data preprocessing file to split video into frames
●	Dataset: https://www.kaggle.com/c/deepfake-detection-challenge

