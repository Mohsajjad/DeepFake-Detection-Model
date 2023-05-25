

import numpy as np
import os
from dash import Dash, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import cv2
from binascii import a2b_base64
import urllib
import dash_player as dp

def Header(name, app):
    title = html.H1(name, style={"margin-top": 21, 'textAlign': 'center' })
    return dbc.Row([dbc.Col(title, md=24)])

# Instantiate our App and incorporate BOOTSTRAP theme stylesheet
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])


    

# Build the layout to define what will be displayed on the page
app.layout = dbc.Container([
     Header("Deepfore-Deepfake Detection", app),
        html.Hr(),
   
     
    dbc.Row([
        dbc.Col([
            html.Label("Upload / Drag and drop an image ")
            ], width=6),
        
        ]),
    
     dbc.Row([
         dbc.Col([
             html.Div(
                 children=[
                   dcc.Upload(
                      id='upload-image',                                            
                         children=html.Div( html.A('Select file')
                             ),
                         style={
                             'width': '50%',
                             'height': '60px',
                             'lineHeight': '60px',
                             'borderWidth': '1px',
                             'borderStyle': 'dashed',
                             'borderRadius': '5px',
                             'textAlign': 'center',
                             'margin': '10px'
                             },
                         multiple=True
                            )
                         ]
                     ),  
            html.Div(
                children=[                 
                    html.Div(id='output-image-upload'),
                        ]),
                    
                    
             ],width=6),
         dbc.Col([
            dbc.Button("Run to detect", size="lg",className="me-md-2",id="run",color="primary"),
            html.Div(id='pie-chart', children=[])], width=4, style={'height' : "300px"}),
         

        ]),
    html.Hr(),

    dbc.Row(html.H1(id='real-or-fake'), className='text-center', style={"font-weight": "bold", 'font_size': '500px'} ),
     
])


image_path="image_folder/input_image" 


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),)
def upload_image(contents, filename):
    
    for f in os.listdir("image_folder/"):
       os.remove(os.path.join("image_folder/",f))
    
    if contents is not None: 
        response = urllib.request.urlopen(contents[0])
        with open(image_path+"."+filename[0].split(".", 1)[1], 'wb') as f:
            f.write(response.file.read())
        if filename[0].split('.')[-1] == 'mp4':
            return html.Div([
                html.H5(filename),
                #html.Video(src=image_path+'/'+filename[0], controls=True),
                dp.DashPlayer(url=image_path+'/'+filename[0], controls=True, width="100%",
                            height="250px",),
                html.Hr(),
                ])
        else:
            return html.Div([
                html.H5(filename),
                html.Img(src=contents, width='100%', height='250px'),
                html.Hr(),
                ])
    
        
            


@app.callback(
        [Output("real-or-fake", "children"),
         Output('pie-chart', 'children')],
        Input("run", "n_clicks"))

def update_output(n_clicks):
    import cv2
    import pandas as pd
    import numpy as np
    import os
    from matplotlib.pyplot import imread
    from matplotlib.pyplot import imshow
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    
    if n_clicks is None or n_clicks==0:
        return no_update
    for f in os.listdir("image_folder/"):
       input_image=os.path.join("image_folder/",f)
    
    
    NUM_CLASSES = 2
    IMG_SIZE = 224
    size = (IMG_SIZE, IMG_SIZE)
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #tf.compat.v1.disable_eager_execution()
    # Using model without transfer learning
    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )
    model.load_weights('training_weight2/cp.ckpt').expect_partial()
    
    if input_image.split('.')[-1] == 'jpg' or input_image.split('.')[-1] == 'jpeg' or input_image.split('.')[-1] == 'png' or input_image.split('.')[-1] == 'webp':
        img_path = input_image

        #img = image.load_img(img_path, target_size=(224, 224))
        #x = img.img_to_array(img)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))

        x5 = np.expand_dims(img, axis=0)
        x5 = preprocess_input(x5)

        print('Input image shape:', x5.shape)

        my_image = imread(img_path)
        imshow(my_image)
        
        preds=model.predict(x5)
        labels = np.argmax(preds, axis=-1)
        
        #Converting the output into readable classes (Real or Fake)
        class_preds = {'REAL': [], 'FAKE': []}
        class_conf = []
        for i, j in zip(labels, preds):
            if i == 1:
                class_preds['REAL'].append(j[1])
            else:
                class_preds['FAKE'].append(j[0])
        
        #Determing the confidence level of each class
        real_conf = 0
        fake_conf = 0
        if len(class_preds['REAL']) > 0:
            real_conf += round(int(class_preds['REAL'][0]*100),2)
        elif len(class_preds['FAKE']) > 0:
            fake_conf += round(int(class_preds['FAKE'][0]*100),2)
        
        fig = px.pie(values=[real_conf, fake_conf], names=['Real', 'Fake'], hole=.3,)
        fig.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)'
                        })
        
        #Printing the result
        for i in labels:
            if i == 1:
                return f'{real_conf}% confident that this is a real image', dcc.Graph(id='display-chart', figure=fig)
            else:
                return f'{fake_conf}% confident that this is a fake image', dcc.Graph(id='display-chart', figure=fig)
            
    #Begin testing for video files            
    elif input_image.split('.')[-1] == 'mp4':
        #Split the video into frames
        t_images = []
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        gap = 4 #setting a gap variable to skip frames and print every 4th frame
        name = input_image.split('/')[-1].split('.')[0]
        if input_image.endswith('mp4'):
            capture = cv2.VideoCapture(input_image)
            frameNr = 0 #setting the frame counter
            while (True):
                success, frame = capture.read()
                if success == False:
                    capture.release()
                    break

                elif frameNr == 0 or frameNr % gap == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    try:
                        for (x,y,w,h) in faces:
                            roi_color = frame[y:y+h, x:x+w]  #y-10:y + h+10, x-10:x + w +10
                        resized = cv2.resize(roi_color, (224,224))
                        t_images.append(resized)
                    except:
                        pass
                    
                frameNr = frameNr+1
        
        
        #Converting the images into [0,255] range
        t_images = np.array(t_images)
        t_images = t_images.astype('float32') / 255.0
        print(t_images.shape)
        #Predicting the output
        preds=model.predict(t_images)
        labels = np.argmax(preds, axis=-1)
        
        #Converting the output into readable classes (Real or Fake)
        
        class_preds = {'REAL': [], 'FAKE': []}
        class_conf = []
        for i, j in zip(labels, preds):
            if i == 1:
                class_preds['REAL'].append(j[1])
            else:
                class_preds['FAKE'].append(j[0])
        
        #Checking for confidence level for each class determined
        real_conf = 0
        fake_conf = 0
        if len(class_preds['REAL']) > 0:
            real_conf += round(len(class_preds['REAL'])/(len(class_preds['REAL'])+len(class_preds['FAKE']))*100,2)  
        if len(class_preds['FAKE']) > 0:
            fake_conf += round(len(class_preds['FAKE'])/(len(class_preds['REAL'])+len(class_preds['FAKE']))*100,2)
            
        fig = px.pie(values=[real_conf, fake_conf], names=['Real', 'Fake'],hole=.3,)
        fig.update_layout({
                'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)'
                        })
        fig.show()
        
        #Printing the resul
        if len(class_preds['REAL']) > len(class_preds['FAKE']):
            return f'{real_conf}% confident that this video is real', dcc.Graph(id='display-chart', figure=fig)
        elif len(class_preds['REAL']) < len(class_preds['FAKE']):
            return f'{fake_conf}% confident that this video is fake', dcc.Graph(id='display-chart', figure=fig)
        elif len(class_preds['REAL']) == len(class_preds['FAKE']):
            return 'COULD NOT DETECT', dcc.Graph(id='display-chart', figure=fig)
    


  
  




# Run the App
if __name__ == '__main__':
    #app.run_server(debug = True)
    app.run_server(host="0.0.0.0",port=8080)

    
