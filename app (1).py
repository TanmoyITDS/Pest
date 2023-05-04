import tensorflow as tf
from tensorflow import keras
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
from google.colab.patches import cv2_imshow


from keras.models import load_model
from PIL import Image
from PIL import Image, ImageOps
import streamlit as st
import numpy as np

model = load_model("/content/drive/MyDrive/tf/keras_model.h5", compile=False) 
class_names = open("/content/drive/MyDrive/tf/labels.txt", "r").readlines()

st.write("""
        #Crop Pest Identification
        """)

st.write("This is a sample image Classification web app to predict crop pest name")
file = st.file_uploader("Upload an Image File", type = ['jpg', 'png'])

def import_and_predict(image, model):
    
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.BICUBIC)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:]
        confidence_score = prediction[0][index]
        
        return class_name #, confidence_score
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    st.write("It is a ",prediction)
